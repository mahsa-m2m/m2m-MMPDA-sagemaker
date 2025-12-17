from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchaudio
import cv2
import numpy as np
import os
import sys
import random
from pathlib import Path
import subprocess
import tempfile
import warnings
import logging
import traceback 
import mediapipe as mp
from torch.cuda.amp import autocast, GradScaler
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


# SageMaker paths (with fallbacks for local testing)
CHECKPOINT_DIR = os.environ.get('SM_CHECKPOINT_DIR', './checkpoints')
# CHECKPOINT_DIR = '/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/logs'
MODEL_DIR = os.environ.get('SM_MODEL_DIR', './model')
# MODEL_DIR = '/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/logs'
LOG_DIR = os.environ.get('SM_CHECKPOINT_DIR', './logs')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==================== WARNING SUPPRESSION ====================

# Suppress Python warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# Use scipy resampler instead of resampy
os.environ['LIBROSA_RESAMPLE_BACKEND'] = 'scipy'

import librosa
from models_comp.fusion_model import FusionModule, LightweightFusionModule, MinimalFusionModule
from utils import AvgrageMeter, performances
import DALoss
import DANetwork

def install_system_dependencies():
    """
    Installs system-level dependencies required for Video/Audio processing
    and OpenCV on standard SageMaker containers.
    """
    print(" Checking system dependencies...")
    try:
        # We use ffmpeg as a proxy to check if we've already installed packages
        subprocess.check_call(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(" ✅ System dependencies appear to be installed.")
    except (OSError, subprocess.CalledProcessError):
        print(" 🔧 System dependencies missing. Installing via apt-get...")
        try:
            # Install ALL required libraries
            # 1. ffmpeg: Video processing
            # 2. libsndfile1: Audio loading (librosa)
            # 3. libgl1 & libglib2.0-0: OpenCV graphics dependencies
            cmd = 'apt-get update -y && apt-get install -y ffmpeg libsndfile1 libgl1 libglib2.0-0'
            
            subprocess.check_call(cmd, shell=True)
            print("   ✅ All system dependencies installed successfully.")
        except Exception as e:
            print(f"   ❌ Failed to install dependencies: {e}")

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class PreprocessedVideoDataset(Dataset):
    """
    Dataset for loading pre-extracted .pt feature files.
    
    Handles mapping from CSV (raw video paths) to .pt files (preprocessed features).
    
    CSV Example:
        s3://bucket/dataset/video/deceptive/video.mkv, deceptive
    
    Maps to:
        {data_root}/video.pt
        e.g., /local/path/train/video.pt
    
    Expected .pt file structure:
    {
        'vision_behaviour': torch.Tensor [T, 50],
        'vision_face': torch.Tensor [3, T, H, W] (uint8),
        'audio_mel': torch.Tensor [3, 128, time_steps],
        'audio_wave': torch.Tensor [audio_length],
        'label': torch.Tensor (long),
        'videoname': str
    }
    """

    def __init__(self, csv_file=None, data_root=None, mode='train'):
        """
        Args:
            csv_file: Path to CSV with columns [s3_path, label]
                     e.g., train.csv with paths like:
                     s3://bucket/dataset/video/deceptive/video.mkv, deceptive
            
            data_root: Root directory containing the .pt files
                      This is where your EXTRACTED features are stored.
                      e.g., /opt/ml/input/data/train/
                      or s3://bucket/dataset/video/precomputed_features/train/
            
            mode: 'train', 'val', or 'test'
        
        Usage:
            # Training
            train_dataset = PreprocessedVideoDataset(
                csv_file='/path/to/train.csv',
                data_root='/local/mount/train/',  # Contains .pt files
                mode='train'
            )
            
            # Validation
            val_dataset = PreprocessedVideoDataset(
                csv_file='/path/to/validation.csv',
                data_root='/local/mount/validation/',  # Contains .pt files
                mode='val'
            )
        """
        self.data_root = data_root
        self.mode = mode
        self.samples = []
        
        # Label mapping (same as your extraction code)
        self.label_map = {'truthful': 0, 'deceptive': 1, 'truth': 0, 'lie': 1}
        
        # Load data
        if csv_file is not None and os.path.exists(csv_file):
            self._load_from_csv(csv_file, data_root)
        elif data_root is not None:
            self._load_from_directory(data_root)
        else:
            raise ValueError("Either csv_file or data_root must be provided")
        
        print(f"✅ Loaded {len(self.samples)} preprocessed samples for {mode} set")
        if len(self.samples) == 0:
            raise ValueError(f"❌ No .pt files found! Check data_root: {data_root}")

    def _load_from_csv(self, csv_file, data_root):
        """
        Load preprocessed .pt files based on CSV mappings.
        
        CSV format: [s3_path, label]
        Example row: s3://bucket/dataset/video/deceptive/video.mkv, deceptive
        
        Mapping logic:
        1. Extract filename from S3 path: video.mkv
        2. Remove extension: video
        3. Add .pt extension: video.pt
        4. Look for file in data_root: {data_root}/video.pt
        """
        print(f"📄 Loading from CSV: {csv_file}")
        print(f"📂 Looking for .pt files in: {data_root}")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Handle column naming variations
            if 's3_path' not in df.columns and 'path' not in df.columns:
                # No header detected
                df = pd.read_csv(csv_file, header=None)
                df.columns = ['path', 'label']
            else:
                # Normalize column names
                cols = list(df.columns)
                rename_map = {c: 'path' for c in cols if 'path' in c.lower()}
                rename_map.update({c: 'label' for c in cols if 'label' in c.lower()})
                df.rename(columns=rename_map, inplace=True)
        
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return
        
        valid_count = 0
        missing_count = 0
        missing_examples = []
        
        for idx, row in df.iterrows():
            # Get original video path and label
            raw_path = str(row['path']).strip()
            label_raw = str(row['label']).strip().lower()
            
            # Map label
            if label_raw not in self.label_map:
                if missing_count < 3:
                    print(f"⚠️ Unknown label '{label_raw}' for {raw_path}")
                continue
            label = self.label_map[label_raw]
            
            # ========================================
            # PATH MAPPING: S3 video → local .pt file
            # ========================================
            # Input:  s3://bucket/dataset/video/deceptive/TTTT_228.mkv
            # Output: /local/train/TTTT_228.pt
            
            # Step 1: Extract filename from S3 path
            filename = os.path.basename(raw_path)  # "TTTT_228.mkv"
            
            # Step 2: Remove video extension and add .pt
            file_base = os.path.splitext(filename)[0]  # "TTTT_228"
            pt_filename = f"{file_base}.pt"            # "TTTT_228.pt"
            
            # Step 3: Construct full local path to .pt file
            # The .pt files are stored FLAT in data_root (no subdirectories)
            pt_path = os.path.join(data_root, pt_filename)
            
            # Step 4: Verify file exists
            if os.path.exists(pt_path):
                self.samples.append({
                    'pt_path': pt_path,
                    'label': label,
                    'filename': filename,
                    'original_path': raw_path
                })
                valid_count += 1
            else:
                if missing_count < 5:  # Show first 5 missing files
                    missing_examples.append({
                        'csv_path': raw_path,
                        'expected_pt': pt_path
                    })
                missing_count += 1
        
        # Report results
        print(f"✅ Found {valid_count} valid .pt files")
        
        if missing_count > 0:
            print(f"\n⚠️ WARNING: {missing_count} .pt files not found!")
            print("   This usually means:")
            print("   1. Feature extraction didn't complete for all videos")
            print("   2. data_root path is incorrect")
            print("   3. CSV references videos that weren't extracted")
            
            if missing_examples:
                print("\n   Examples of missing files:")
                for ex in missing_examples[:3]:
                    print(f"   CSV entry: {ex['csv_path']}")
                    print(f"   Expected:  {ex['expected_pt']}")
                    print()

    def _load_from_directory(self, data_root):
        """
        Load all .pt files from directory (fallback method).
        
        Useful when you don't have a CSV, just a folder of .pt files.
        
        Assumes structure:
        data_root/
            video1.pt
            video2.pt
            ...
        
        Labels are inferred from filenames containing 'deceptive'/'truthful'.
        """
        print(f"📂 Loading all .pt files from: {data_root}")
        
        data_root = Path(data_root)
        
        if not data_root.exists():
            print(f"❌ Directory does not exist: {data_root}")
            return
        
        pt_files = list(data_root.glob('*.pt'))
        
        if len(pt_files) == 0:
            print(f"⚠️ No .pt files found in {data_root}")
            return
        
        for pt_file in pt_files:
            # Try to infer label from filename
            filename = pt_file.stem.lower()
            
            if 'deceptive' in filename or 'lie' in filename:
                label = 1
            elif 'truthful' in filename or 'truth' in filename:
                label = 0
            else:
                # Default to 0 if unclear
                print(f"⚠️ Cannot infer label for {pt_file.name}, defaulting to 0")
                label = 0
            
            self.samples.append({
                'pt_path': str(pt_file),
                'label': label,
                'filename': pt_file.name,
                'original_path': str(pt_file)
            })
        
        print(f"✅ Found {len(pt_files)} .pt files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load a preprocessed sample from disk.
        
        Returns:
            dict with keys:
                - vision_behaviour: [T, 50] float32
                - vision_face: [3, T, H, W] float32 (normalized)
                - audio_mel: [3, 128, time_steps] float32
                - audio_wave: [audio_length] float32
                - label: long
                - videoname: str
        """
        sample_info = self.samples[idx]
        
        try:
            # Load the .pt file
            data = torch.load(sample_info['pt_path'], map_location='cpu')
            
            # Extract components
            vision_behaviour = data['vision_behaviour']  # [T, 50]
            vision_face = data['vision_face']             # [3, T, H, W] uint8
            audio_mel = data['audio_mel']                 # [3, 128, time_steps]
            audio_wave = data['audio_wave']               # [audio_length]
            label = data['label']                          # scalar
            videoname = data.get('videoname', sample_info['filename'])
            
            # ====== POST-PROCESSING ======
            
            # 1. Convert face from uint8 [0-255] to float32 [0-1]
            vision_face = vision_face.float() / 255.0
            
            # 2. Normalize face images (ImageNet stats)
            # Shape: [3, T, H, W]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
            vision_face = (vision_face - mean) / std
            
            # 3. Ensure correct dtypes
            vision_behaviour = vision_behaviour.float()
            audio_mel = audio_mel.float()
            audio_wave = audio_wave.float()
            label = label.long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)
            
            return {
                'vision_behaviour': vision_behaviour,
                'vision_face': vision_face,
                'audio_mel': audio_mel,
                'audio_wave': audio_wave,
                'label': label,
                'videoname': videoname
            }
        
        except Exception as e:
            print(f"❌ Error loading {sample_info['pt_path']}: {e}")
            # Return a dummy sample to avoid breaking training
            return self._get_dummy_sample(sample_info['label'])
    
    def _get_dummy_sample(self, label):
        """
        Fallback dummy sample in case of loading errors.
        Matches the expected output format.
        """
        return {
            'vision_behaviour': torch.zeros(64, 50, dtype=torch.float32),
            'vision_face': torch.zeros(3, 64, 160, 160, dtype=torch.float32),
            'audio_mel': torch.zeros(3, 128, 501, dtype=torch.float32),  # 80000//160 + 1
            'audio_wave': torch.zeros(80000, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'videoname': 'dummy_sample'
        }

def compute_class_weights(dataset):
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        dataset: PreprocessedVideoDataset instance
    
    Returns:
        torch.Tensor: Class weights [weight_class_0, weight_class_1]
    """
    labels = [sample['label'] for sample in dataset.samples]
    
    # Count occurrences
    unique, counts = np.unique(labels, return_counts=True)
    
    # Compute inverse frequency weights
    total = len(labels)
    weights = total / (len(unique) * counts)
    
    print(f"📊 Class Distribution:")
    for cls, count, weight in zip(unique, counts, weights):
        class_name = 'Truthful' if cls == 0 else 'Deceptive'
        print(f"   Class {cls} ({class_name}): {count} samples (weight: {weight:.3f})")
    
    return torch.FloatTensor(weights)

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Better than weighted CE for imbalanced datasets
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights [weight_class0, weight_class1]
        self.gamma = gamma  # Focusing parameter (default: 2)
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, args, scaler):
    """Train for one epoch"""
    model.train()

    loss_global = AvgrageMeter()
    loss_vl = AvgrageMeter()
    loss_face = AvgrageMeter()
    loss_al = AvgrageMeter()

    correct = 0
    total = 0

    aux_weight = 0.2

    for i, sample_batched in enumerate(dataloader):
        # print(f"Batch {i+1} loaded", flush=True) 
        # Get data
        vision_behaviour = sample_batched['vision_behaviour'].to(device)  # (B, T, 64)
        vision_face = sample_batched['vision_face'].to(device)  # (B, C, T, H, W)
        audio_mel = sample_batched['audio_mel'].to(device)  # (B, 3, n_mels, time)
        audio_wave = sample_batched['audio_wave'].to(device)  # (B, audio_length)
        labels = sample_batched['label'].to(device)  # (B,)
        
        if torch.isnan(audio_mel).any() or torch.isinf(audio_mel).any():
            print(f"⚠️ Warning: Batch {i} contains NaN/Inf audio. Skipping.")
            continue
            
        # Clamp audio to safe range (e.g., -100 to 100) to prevent log(0) explosions
        audio_mel = torch.clamp(audio_mel, min=-100.0, max=100.0)
        
        # Forward pass
        optimizer.zero_grad()

        with autocast():
            # Call fusion model with all 4 modalities
            fused_logit, vl_logit, face_logit, al_logit, feat_list = model(
                vision_behaviour, vision_face, audio_mel, audio_wave
            )

            # Calculate losses
            global_loss = criterion(fused_logit, labels)

            if vl_logit is not None:
                vl_loss = criterion(vl_logit, labels)
                face_loss = criterion(face_logit, labels)
                al_loss = criterion(al_logit, labels)

                # Focus on Main, treat others as hints (0.2)
                loss = global_loss + aux_weight * (vl_loss + face_loss + al_loss)
                # loss = global_loss + vl_loss + face_loss + al_loss
            else:
                loss = global_loss
                vl_loss = torch.tensor(0.0)
                face_loss = torch.tensor(0.0)
                al_loss = torch.tensor(0.0)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Warning: Batch {i} loss is NaN. Skipping optimizer step.")
            # Clear cache if this happens
            torch.cuda.empty_cache()
            continue

        # BACKWARD PASS WITH SCALER
        # Scales loss to prevent underflow in float16
        scaler.scale(loss).backward()
        # loss.backward()

        # Gradient clipping (Unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

        # Statistics
        n = vision_face.size(0)
        loss_global.update(global_loss.item(), n)
        if vl_logit is not None:
            loss_vl.update(vl_loss.item(), n)
            loss_face.update(face_loss.item(), n)
            loss_al.update(al_loss.item(), n)

        _, predicted = torch.max(fused_logit.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % args.echo_batches == 0:
            print(f'Epoch [{epoch}], Step [{i + 1}/{len(dataloader)}], '
                  f'Loss_global: {loss_global.avg:.4f}, Loss_vl: {loss_vl.avg:.4f}, '
                  f'Loss_face: {loss_face.avg:.4f}, Loss_al: {loss_al.avg:.4f}, '
                  f'Acc: {100 * correct / total:.2f}%')

    epoch_acc = 100 * correct / total
    return loss_global.avg, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model with Accuracy and F1 Score"""
    model.eval()

    loss_meter = AvgrageMeter()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for sample_batched in dataloader:
            vision_behaviour = sample_batched['vision_behaviour'].to(device)
            vision_face = sample_batched['vision_face'].to(device)
            audio_mel = sample_batched['audio_mel'].to(device)
            audio_wave = sample_batched['audio_wave'].to(device)
            labels = sample_batched['label'].to(device)

            # Forward pass
            fused_logit, _, _, _, _ = model(
                vision_behaviour, vision_face, audio_mel, audio_wave
            )

            loss = criterion(fused_logit, labels)
            loss_meter.update(loss.item(), vision_face.size(0))

            probs = F.softmax(fused_logit, dim=1)
            _, predicted = torch.max(fused_logit.data, 1)

            # Move to CPU for sklearn metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())

    # Calculate Accuracy
    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    accuracy = 100 * correct / len(all_labels)

    # Calculate F1 Score
    # average='weighted' accounts for class imbalance (Truth > Deceptive)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Return f1 as well
    return loss_meter.avg, accuracy, f1, all_preds, all_labels, all_scores

def save_checkpoint(model, optimizer, epoch, current_step, best_val_acc, scheduler_warmup, scheduler_cosine, warmup_steps, min_val_loss):
    """Save checkpoint for spot instance recovery."""
    checkpoint = {
        'epoch': epoch,
        'current_step': current_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_warmup_state': scheduler_warmup.state_dict(),
        'scheduler_cosine_state': scheduler_cosine.state_dict(),
        'warmup_steps': warmup_steps,
        'best_val_acc': best_val_acc,
        'min_val_loss': min_val_loss
    }
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'latest.pt'))
    print(f"✅ Checkpoint saved: epoch {epoch + 1}")

def load_checkpoint(model, optimizer, scheduler_warmup, scheduler_cosine, device):
    """Load checkpoint if exists (after spot interruption)."""
    path = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    min_val_loss = float('inf')
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler_warmup.load_state_dict(ckpt['scheduler_warmup_state'])
        scheduler_cosine.load_state_dict(ckpt['scheduler_cosine_state'])
        min_val_loss = ckpt.get('min_val_loss', float('inf'))

        print(f"🔄 Resumed from epoch {ckpt['epoch'] + 1}")
        return ckpt['epoch'] + 1, ckpt['current_step'], ckpt['best_val_acc'], min_val_loss
    return 0, 0, 0.0, min_val_loss

# def compute_class_weights(dataset):
#     """
#     Fast version - directly access labels without loading full samples
#     Only use if the dataset has a direct label access method
#     """
#     print(f"\n{'='*60}")
#     print(f"📊 Computing Class Weights (Fast Mode)...")
#     print(f"{'='*60}")
    
#     # Check if dataset has direct label access
#     if hasattr(dataset, 'labels'):
#         labels = dataset.labels
#         print(f"  ✅ Using pre-loaded labels from dataset")
#     else:
#         print(f"  ⚠️ No direct label access, falling back to full loading")
#         return compute_class_weights(dataset)
    
#     class_counts = np.bincount(labels)
#     total = sum(class_counts)
#     num_classes = len(class_counts)
    
#     print(f"\n📊 Dataset Statistics:")
#     print(f"  Total samples: {total}")
#     print(f"  Class 0 (Truth): {class_counts[0]} samples ({100*class_counts[0]/total:.1f}%)")
#     print(f"  Class 1 (Lie):   {class_counts[1]} samples ({100*class_counts[1]/total:.1f}%)")
#     print(f"  Imbalance ratio: {max(class_counts)/min(class_counts):.2f}:1")
    
#     # Compute inverse frequency weights
#     weights = total / (num_classes * class_counts)
#     weights = weights / weights.sum() * num_classes
    
#     print(f"  Class weights: [Truth: {weights[0]:.3f}, Lie: {weights[1]:.3f}]")
#     print(f"{'='*60}\n")
    
#     return torch.FloatTensor(weights)

# def main(args):

#     # Setup
#     setup_seed(42)
#     device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

#     # Create log directory
#     # os.makedirs(args.log, exist_ok=True)
#     # log_file = open(os.path.join(args.log, 'training_log.txt'), 'a')  # 'a' for resume
#     log_file = open(os.path.join(LOG_DIR, 'training_log.txt'), 'a', buffering=1)

#     print(f"Using device: {device}")
#     print(f"Arguments: {args}")
#     log_file.write(f"Arguments: {args}\n")

#     # --- Training Set ---
#     if args.train_list:
#         print(f"\n  Initializing Training Dataset from CSV: {args.train_list}")
#         if not os.path.exists(args.train_list):
#             raise FileNotFoundError(f"Training CSV not found: {args.train_list}")
            
#         train_dataset = VideoDeceptionDataset(
#             csv_file=args.train_list,
#             data_root=args.train_root,
#             num_frames=args.num_frames,
#             frame_size=(args.frame_height, args.frame_width),
#             mode='train'
#         )
#     elif args.train_root:
#         print(f"\n🏗️  Initializing Training Dataset from Folder: {args.train_root}")
#         train_dataset = VideoDeceptionDataset(
#             data_root=args.train_root,
#             num_frames=args.num_frames,
#             frame_size=(args.frame_height, args.frame_width),
#             mode='train'
#         )
#     else:
#         raise ValueError("❌ Error: You must provide either --train_list (CSV) or --train_root (Folder)")

#     # --- Validation Set ---
#     if args.val_list:
#         print(f"\n🏗️  Initializing Validation Dataset from CSV: {args.val_list}")
#         if not os.path.exists(args.val_list):
#             print(f"⚠️ Warning: Validation CSV not found at {args.val_list}. skipping validation.")
#             val_dataset = None
#         else:
#             val_dataset = VideoDeceptionDataset(
#                 csv_file=args.val_list,
#                 data_root=args.train_root,
#                 num_frames=args.num_frames,
#                 frame_size=(args.frame_height, args.frame_width),
#                 mode='val'
#             )
#     elif args.val_root:
#         val_dataset = VideoDeceptionDataset(
#             data_root=args.val_root,
#             num_frames=args.num_frames,
#             frame_size=(args.frame_height, args.frame_width),
#             mode='val'
#         )
#     else:
#         print("⚠️ No validation data provided.")
#         val_dataset = None
#     # if args.train_root:
#     #     train_dataset = VideoDeceptionDataset(
#     #         csv_file=args.train_list,
#     #         data_root=args.train_root,
#     #         num_frames=args.num_frames,
#     #         frame_size=(args.frame_height, args.frame_width),
#     #         audio_length=args.audio_length,
#     #         sample_rate=args.sample_rate,
#     #         n_mels=args.n_mels,
#     #         mode='train'
#     #     )
#     # else:
#     #     train_dataset = VideoDeceptionDataset(
#     #         annotation_file=args.train_list,
#     #         num_frames=args.num_frames,
#     #         frame_size=(args.frame_height, args.frame_width),
#     #         audio_length=args.audio_length,
#     #         sample_rate=args.sample_rate,
#     #         n_mels=args.n_mels,
#     #         mode='train'
#     #     )

#     # if args.val_root:
#     #     val_dataset = VideoDeceptionDataset(
#     #         csv_file=args.train_list,
#     #         data_root=args.val_root,
#     #         num_frames=args.num_frames,
#     #         frame_size=(args.frame_height, args.frame_width),
#     #         audio_length=args.audio_length,
#     #         sample_rate=args.sample_rate,
#     #         n_mels=args.n_mels,
#     #         mode='val'
#     #     )
#     # else:
#     #     val_dataset = VideoDeceptionDataset(
#     #         annotation_file=args.val_list,
#     #         num_frames=args.num_frames,
#     #         frame_size=(args.frame_height, args.frame_width),
#     #         audio_length=args.audio_length,
#     #         sample_rate=args.sample_rate,
#     #         n_mels=args.n_mels,
#     #         mode='val'
#     #     )

#     print("============= Class Weight ==========")
#     class_weights = compute_class_weights(train_dataset)
#     print(class_weights)
   
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batchsize,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batchsize,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True
#     )

#     # Add device to args for model
#     args.device = device

#     print(f"\n  Creating Model: {args.model_arch.upper()}")

#     if args.model_arch == 'normal':
#         # The robust MULT transformer model (Best for 100k data)
#         model = FusionModule(args)
#     elif args.model_arch == 'lite':
#         # The intermediate model
#         model = LightweightFusionModule(args)
#     elif args.model_arch == 'minimal':
#         # The very small model (Best for debugging/small data)
#         model = MinimalFusionModule(args)
#     else:
#         raise ValueError(f"❌ Unknown model architecture: {args.model_arch}")

#     # ###### FREEZING MODEL
#     # for param in model.audio_model.parameters():
#     #     param.requires_grad = False
#     # for param in model.face_model.parameters():
#     #     param.requires_grad = False

#     model = model.to(device)

#     if torch.cuda.device_count() > 1:
#         print(f"\n🚀 Detected {torch.cuda.device_count()} GPUs! Activating DataParallel.")
#         model = nn.DataParallel(model)

#     print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
#     log_file.write(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")

#     ### Focal loss
#     # criterion = FocalLoss(
#     #     alpha=class_weights.to(device),
#     #     gamma=0.3  # Focusing parameter 2.0
#     # )

#     # Loss and optimizer
#     soft_weights = torch.tensor([1.0, 1.91]).to(device)
#     criterion = nn.CrossEntropyLoss(weight=soft_weights)
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

#     # Learning rate scheduler
#     warmup_epochs = 1
#     steps_per_epoch = len(train_loader)
#     total_steps = args.max_epochs * steps_per_epoch
#     warmup_steps = warmup_epochs * steps_per_epoch

#     def lr_lambda(current_step):
#         if current_step < warmup_steps:
#             return float(current_step) / float(max(1, warmup_steps))
#         else:
#             return 1.0

#     scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
#     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
#     )

#     # LOAD CHECKPOINT IF RESUMING
#     start_epoch, current_step, best_val_acc, min_val_loss = load_checkpoint(
#         model, optimizer, scheduler_warmup, scheduler_cosine, device
#     )

#     scaler = GradScaler()
    
#     val_loss = min_val_loss 

#     # Training loop
#     for epoch in range(start_epoch, args.max_epochs):
#         print(f"\n{'=' * 50}")
#         print(f"Epoch {epoch + 1}/{args.max_epochs}")
#         print(f"{'=' * 50}")

#         # if epoch == 2:
#         #     if isinstance(model, nn.DataParallel):
#         #         actual_model = model.module
#         #     else:
#         #         actual_model = model
#         #     print(" Unfreezing encoders...")
#         #     for param in actual_model.audio_model.parameters():
#         #         param.requires_grad = True
#         #     for param in actual_model.face_model.parameters():
#         #         param.requires_grad = True

#         # Train
#         train_loss, train_acc = train_one_epoch(
#             model, train_loader, criterion, optimizer, device, epoch + 1, args, scaler
#         )

#         print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
#         log_file.write(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n")

#         # Update learning rate
#         for _ in range(steps_per_epoch):
#             if current_step < warmup_steps:
#                 scheduler_warmup.step()
#             else:
#                 scheduler_cosine.step()
#             current_step += 1

#         # Validate
#         if (epoch + 1) % args.val_interval == 0:
#             # val_loss, val_acc, val_preds, val_labels, val_scores = validate(
#             #     model, val_loader, criterion, device
#             # )
#             val_loss, val_acc, val_f1, val_preds, val_labels, val_scores = validate(
#                 model, val_loader, criterion, device
#             )

#             # print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
#             print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
#             log_file.write(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}\n")
#             # print(f"Epoch [{epoch + 1}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
#             # log_file.write(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

#             # Save best model - accuracy
#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc

#                 # Unwrap model before saving Best Model
#                 if isinstance(model, nn.DataParallel):
#                     model_to_save = model.module
#                 else:
#                     model_to_save = model
                
#                 # filename = f'best_model_epoch_{epoch + 1}.pt'
#                 # filename = 'best_model_acc.pt'
#                 filename = f'best_model_acc_{int(val_acc)}_ep{epoch+1}.pt'
#                 save_path = os.path.join(CHECKPOINT_DIR, filename)

#                 # Save to checkpoint dir (synced to S3)
#                 # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#                 torch.save({
#                     'epoch': epoch + 1,
#                     'model_state_dict': model_to_save.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'best_acc': best_val_acc,
#                     'val_loss': val_loss,
#                     'args': vars(args)
#                 }, save_path)
                
#                 # # Also save to log dir
#                 # torch.save({
#                 #     'epoch': epoch + 1,
#                 #     'model_state_dict': model_to_save.state_dict(),
#                 #     'optimizer_state_dict': optimizer.state_dict(),
#                 #     'best_acc': best_val_acc,
#                 # }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
#                 print(f"🏆 Saved best model with accuracy: {best_val_acc:.2f}%")
#                 log_file.write(f"Saved best model with accuracy: {best_val_acc:.2f}%\n")
            
#             # Save best model - val loss
#             if val_loss < min_val_loss:
                
#                 min_val_loss = val_loss

#                 # Save filename
#                 # filename = 'best_model_loss.pt'
#                 filename = f'best_model_loss_ep{epoch+1}_acc{int(val_acc)}.pt'

#                 save_path = os.path.join(CHECKPOINT_DIR, filename)

#                 # Unwrap model before saving Best Model
#                 if isinstance(model, nn.DataParallel):
#                     model_to_save = model.module
#                 else:
#                     model_to_save = model
                
#                 # Save to checkpoint dir (synced to S3)
#                 # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#                 torch.save({
#                     'epoch': epoch + 1,
#                     'model_state_dict': model_to_save.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'best_acc': best_val_acc,
#                     'min_val_loss': min_val_loss,
#                     'args': vars(args)
#                 }, save_path)
                
#                 print(f" New Lowest Loss: {min_val_loss:.4f} (Saved to {filename})")
#                 log_file.write(f"Saved Lowest Loss model: {min_val_loss:.4f}\n")

#         # Unwrap model before passing to custom save function
#         if isinstance(model, nn.DataParallel):
#             model_for_ckpt = model.module
#         else:
#             model_for_ckpt = model
#         # SAVE CHECKPOINT AFTER EACH EPOCH
#         save_checkpoint(
#             model_for_ckpt, optimizer, epoch, current_step, best_val_acc,
#             scheduler_warmup, scheduler_cosine, warmup_steps, min_val_loss
#         )

#         log_file.flush()

#     # Unwrap model before final save
#     if isinstance(model, nn.DataParallel):
#         final_model_to_save = model.module
#     else:
#         final_model_to_save = model

#     # SAVE FINAL MODEL TO SAGEMAKER OUTPUT PATH
#     # os.makedirs(MODEL_DIR, exist_ok=True)
#     # torch.save(final_model_to_save.state_dict(), os.path.join(MODEL_DIR, 'model.pt'))
    
#     torch.save({
#         'model_state_dict': final_model_to_save.state_dict(),
#         'args': vars(args),
#         'best_acc': best_val_acc,
#         'val_loss': val_loss
#     }, os.path.join(MODEL_DIR, 'model_full.pt'))

#     print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
#     print(f"📦 Model saved to {MODEL_DIR}")
#     log_file.write(f"\nBest validation accuracy: {best_val_acc:.2f}%\n")
#     log_file.close()

def main(args):
    """
    Main training function - Updated to use PreprocessedVideoDataset
    """
    # Setup
    setup_seed(42)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Create log directory
    log_file = open(os.path.join(LOG_DIR, 'training_log.txt'), 'a', buffering=1)

    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    log_file.write(f"Arguments: {args}\n")

    ##### DATASET INITIALIZATION
    # --- Training Set ---
    if args.train_list:
        print(f"\n📂 Initializing Training Dataset from CSV: {args.train_list}")
        if not os.path.exists(args.train_list):
            raise FileNotFoundError(f"Training CSV not found: {args.train_list}")
        
        # Use PreprocessedVideoDataset instead of VideoDeceptionDataset
        train_dataset = PreprocessedVideoDataset(
            csv_file=args.train_list,
            data_root=args.train_root,  # Should point to folder with .pt files
            mode='train'
        )
    elif args.train_root:
        print(f"\n📂 Initializing Training Dataset from Folder: {args.train_root}")
        train_dataset = PreprocessedVideoDataset(
            data_root=args.train_root,
            mode='train'
        )
    else:
        raise ValueError("❌ Error: You must provide either --train_list (CSV) or --train_root (Folder)")

    # --- Validation Set ---
    if args.val_list:
        print(f"\n📂 Initializing Validation Dataset from CSV: {args.val_list}")
        if not os.path.exists(args.val_list):
            print(f"⚠️ Warning: Validation CSV not found at {args.val_list}. Skipping validation.")
            val_dataset = None
        else:
            val_dataset = PreprocessedVideoDataset(
                csv_file=args.val_list,
                data_root=args.val_root,  # Should point to folder with .pt files
                mode='val'
            )
    elif args.val_root:
        print(f"\n📂 Initializing Validation Dataset from Folder: {args.val_root}")
        val_dataset = PreprocessedVideoDataset(
            data_root=args.val_root,
            mode='val'
        )
    else:
        print("⚠️ No validation data provided.")
        val_dataset = None

    ###### CLASS WEIGHTS & DATA LOADERS
    
    print("\n============= Class Weights ==========")
    class_weights = compute_class_weights(train_dataset)
    print(class_weights)
   
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        val_loader = None

    ##### MODEL CREATION
    
    args.device = device
    print(f"\n🏗️  Creating Model: {args.model_arch.upper()}")

    if args.model_arch == 'normal':
        model = FusionModule(args)
    elif args.model_arch == 'lite':
        model = LightweightFusionModule(args)
    elif args.model_arch == 'minimal':
        model = MinimalFusionModule(args)
    else:
        raise ValueError(f"❌ Unknown model architecture: {args.model_arch}")

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"\n🚀 Detected {torch.cuda.device_count()} GPUs! Activating DataParallel.")
        model = nn.DataParallel(model)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    log_file.write(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    ##### LOSS, OPTIMIZER, SCHEDULER
    
    # Loss with class weights
    soft_weights = torch.tensor([1.0, 1.91]).to(device)
    criterion = nn.CrossEntropyLoss(weight=soft_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # Learning rate scheduler
    warmup_epochs = 1
    steps_per_epoch = len(train_loader)
    total_steps = args.max_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return 1.0

    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )

    ##### LOAD CHECKPOINT IF RESUMING
    
    start_epoch, current_step, best_val_acc, min_val_loss = load_checkpoint(
        model, optimizer, scheduler_warmup, scheduler_cosine, device
    )

    scaler = GradScaler()
    val_loss = min_val_loss 

    ##### TRAINING LOOP
    
    for epoch in range(start_epoch, args.max_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        print(f"{'=' * 50}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, args, scaler
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        log_file.write(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n")

        # Update learning rate
        for _ in range(steps_per_epoch):
            if current_step < warmup_steps:
                scheduler_warmup.step()
            else:
                scheduler_cosine.step()
            current_step += 1

        # Validate
        if val_loader is not None and (epoch + 1) % args.val_interval == 0:
            val_loss, val_acc, val_f1, val_preds, val_labels, val_scores = validate(
                model, val_loader, criterion, device
            )

            print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
            log_file.write(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}\n")

            # Save best model - accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                # Unwrap model before saving
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                
                filename = f'best_model_acc_{int(val_acc)}_ep{epoch+1}.pt'
                save_path = os.path.join(CHECKPOINT_DIR, filename)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_val_acc,
                    'val_loss': val_loss,
                    'args': vars(args)
                }, save_path)
                
                print(f"🏆 Saved best model with accuracy: {best_val_acc:.2f}%")
                log_file.write(f"Saved best model with accuracy: {best_val_acc:.2f}%\n")
            
            # Save best model - val loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                filename = f'best_model_loss_ep{epoch+1}_acc{int(val_acc)}.pt'
                save_path = os.path.join(CHECKPOINT_DIR, filename)

                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_val_acc,
                    'min_val_loss': min_val_loss,
                    'args': vars(args)
                }, save_path)
                
                print(f"💎 New Lowest Loss: {min_val_loss:.4f} (Saved to {filename})")
                log_file.write(f"Saved Lowest Loss model: {min_val_loss:.4f}\n")

        # Save checkpoint after each epoch
        model_for_ckpt = model.module if isinstance(model, nn.DataParallel) else model
        save_checkpoint(
            model_for_ckpt, optimizer, epoch, current_step, best_val_acc,
            scheduler_warmup, scheduler_cosine, warmup_steps, min_val_loss
        )

        log_file.flush()

    ##### SAVE FINAL MODEL
    
    final_model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    
    torch.save({
        'model_state_dict': final_model_to_save.state_dict(),
        'args': vars(args),
        'best_acc': best_val_acc,
        'val_loss': val_loss
    }, os.path.join(MODEL_DIR, 'model_full.pt'))

    print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📦 Model saved to {MODEL_DIR}")
    log_file.write(f"\nBest validation accuracy: {best_val_acc:.2f}%\n")
    log_file.close()

if __name__ == "__main__":

    install_system_dependencies()

    try:
        print("🚀 SCRIPT STARTED SUCCESSFULLY", flush=True)
            
        parser = argparse.ArgumentParser(description="Multimodal Deception Detection")

        # Training parameters
        parser.add_argument('--gpu', type=int, default=0, help='GPU id')
        parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
        parser.add_argument('--batchsize', type=int, default=8, help='Batch size')
        parser.add_argument('--max_epochs', type=int, default=30, help='Maximum epochs')
        parser.add_argument('--log', type=str, default='logs', help='Log directory')
        parser.add_argument('--echo_batches', type=int, default=5, help='Print frequency')
        parser.add_argument('--val_interval', type=int, default=1, help='Validation interval')
        parser.add_argument('--num_workers', type=int, default=24, help='Dataloader workers')

        # Dataset parameters
        parser.add_argument('--train_root', type=str, default=None,
                            help='Training data root directory')
        parser.add_argument('--val_root', type=str, default=None,
                            help='Validation data root directory')
    
        # local CSV usage
        parser.add_argument('--train_list', type=str, default='sample/train/output/train.csv',
                            help='Path to the training CSV file generated previously')
        parser.add_argument('--val_list', type=str, default='sample/train/output/validation.csv',
                            help='Path to the validation CSV file generated previously')

        # Video/Audio parameters
        parser.add_argument('--num_frames', type=int, default=64,
                            help='Number of frames (T=64 for behavioral features)')
        parser.add_argument('--frame_height', type=int, default=160,
                            help='Frame height')
        parser.add_argument('--frame_width', type=int, default=160,
                            help='Frame width')
        parser.add_argument('--audio_length', type=int, default=16000,
                            help='Audio sample length')
        parser.add_argument('--sample_rate', type=int, default=16000,
                            help='Audio sample rate')
        parser.add_argument('--n_mels', type=int, default=128,
                            help='Number of mel frequency bins')

        # Model parameters 
        parser.add_argument('--model_arch', type=str, default='normal',
                            help='3 different archs: minimal, lite, normal')
        parser.add_argument('--modalities', type=str, default='vaf',
                            help='Modalities: v=visual, a=audio, f=face')
        parser.add_argument('--v_dim', type=int, default=64) #64
        parser.add_argument('--a_dim', type=int, default=512)
        parser.add_argument('--f_dim', type=int, default=512)
        parser.add_argument('--common_dim', type=int, default=128)
        parser.add_argument('--fusion_type', type=str, default='mult',
                            help='Fusion type: concat/transformer/senet/mult')

        # Parameters for mult fusion
        parser.add_argument('--mult_layer', type=int, default=3)
        parser.add_argument('--attn_dropout', type=float, default=0.1)
        parser.add_argument('--relu_dropout', type=float, default=0.1)
        parser.add_argument('--res_dropout', type=float, default=0.1)
        parser.add_argument('--embed_dropout', type=float, default=0.0)
        parser.add_argument('--attn_mask', action='store_false')

        # Parameters for transformer fusion
        parser.add_argument('--embed_dim', type=int, default=64)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--layers', type=int, default=4)

        # Parameters for senet fusion
        parser.add_argument('--channel', type=int, default=64)
        parser.add_argument('--reduction', type=int, default=16)

        args = parser.parse_args()

    # 🔍 DEBUG: Check if source_dir uploaded correctly
        # print(f"📂 Current Directory contents: {os.listdir('.')}", flush=True)
        # if os.path.exists('models_comp'):
        #     print(f"📂 models_comp contents: {os.listdir('models_comp')}", flush=True)
        # else:
        #     print("❌ ERROR: 'models_comp' folder NOT FOUND in container!", flush=True)

        main(args)
        
        print("✅ SCRIPT FINISHED SUCCESSFULLY", flush=True)
        
        
        # # Validate arguments
        # if args.train_root is None and args.train_list is None:
        #     parser.error("Either --train_root or --train_list must be provided")
        # if args.val_root is None and args.val_list is None:
        #     parser.error("Either --val_root or --val_list must be provided")

    except Exception as e:
        print("\n\n❌ ❌ CRITICAL FAILURE ❌ ❌", flush=True)
        print(str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        
        # # Optional: Write to failure file (SageMaker reads this)
        # with open('/opt/ml/output/failure', 'w') as f:
        #     f.write(str(e))
        #     f.write(traceback.format_exc())
        
        # Re-raise so the job is marked as Failed
        raise


