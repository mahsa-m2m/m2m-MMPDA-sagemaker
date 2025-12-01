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
# ==================== COMPREHENSIVE WARNING SUPPRESSION ====================

# 1. Suppress Python warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# Now import librosa with proper backend (without resampy dependency)
# FIX: Use scipy resampler instead of resampy
os.environ['LIBROSA_RESAMPLE_BACKEND'] = 'scipy'
import librosa

# ============================================================================

# Import your models
# from models_comp.fusion_model import FusionModule
from models_comp.fusion_model import LightweightFusionModule, MinimalFusionModule
from utils import AvgrageMeter, performances
import DALoss
import DANetwork

def install_system_dependencies():
    """
    Installs system-level dependencies required for Video/Audio processing
    and OpenCV on standard SageMaker containers.
    """
    print("⚙️ Checking system dependencies...")
    try:
        # We use ffmpeg as a proxy to check if we've already installed packages
        subprocess.check_call(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   ✅ System dependencies appear to be installed.")
    except (OSError, subprocess.CalledProcessError):
        print("   🔧 System dependencies missing. Installing via apt-get...")
        try:
            # Install ALL required libraries in one go
            # 1. ffmpeg: Video processing
            # 2. libsndfile1: Audio loading (librosa)
            # 3. libgl1 & libglib2.0-0: OpenCV graphics dependencies
            cmd = 'apt-get update -y && apt-get install -y ffmpeg libsndfile1 libgl1 libglib2.0-0'
            
            subprocess.check_call(cmd, shell=True)
            print("   ✅ All system dependencies installed successfully.")
        except Exception as e:
            print(f"   ❌ Failed to install dependencies: {e}")
            # We don't exit here, we let the script try to run anyway, 
            # though it will likely crash on import.

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VideoDeceptionDataset(Dataset):
    """
    Dataset for loading video files and extracting multimodal features.

    Expected directory structure:
    data_root/
        truthful/
            video1.mp4
            video2.avi
            ...
        deceptive/
            video1.mp4
            video2.avi
            ...
    """

    def __init__(self, data_root=None, annotation_file=None,
                 num_frames=50, frame_size=(160, 160),
                 audio_length=16000, sample_rate=16000,
                 n_mels=128, mode='train'):
        """
        Args:
            data_root: Root directory with truthful/deceptive subdirectories
            annotation_file: Path to file with video paths and labels
            num_frames: Number of frames to sample (T=64 for behavioral features)
            frame_size: Target size for face frames (H, W)
            audio_length: Length of audio to sample
            sample_rate: Audio sample rate
            n_mels: Number of mel frequency bins
            mode: 'train', 'val', or 'test'
        """
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mode = mode
        self.video_list = []
        self.labels = []

        # Don't initialize MediaPipe here - will be done per worker process
        self.face_mesh = None
        self.face_mesh_initialized = False
        self.mp_face_mesh = mp.solutions.face_mesh

        # Load from directory structure
        if data_root is not None:
            self._load_from_directory(data_root)
        # Load from annotation file
        elif annotation_file is not None:
            self._load_from_annotation(annotation_file)
        else:
            raise ValueError("Either data_root or annotation_file must be provided")

    def _init_mediapipe(self):
        """
        Initialize MediaPipe Face Mesh
        This needs to be called in each worker process
        """
        if self.face_mesh_initialized:
            return
        
        try:
            print("🔧 Initializing MediaPipe Face Mesh...")
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,  # Use True for video frames
                max_num_faces=3,  # Only detect 1 face per frame
                refine_landmarks=True,  # Get more detailed landmarks
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            )
            self.face_mesh_initialized = True
            print("✅ MediaPipe Face Mesh initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize MediaPipe: {str(e)}")
            import traceback
            traceback.print_exc()
            self.face_mesh = None
            self.face_mesh_initialized = False
    
    def _load_from_directory(self, data_root):
        """Load videos from directory structure"""
        data_root = Path(data_root)

        # Load truthful videos (label 0)
        truthful_dir = data_root / 'truthful'
        if truthful_dir.exists():
            for video_file in truthful_dir.glob('*'):
                if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                    self.video_list.append(str(video_file))
                    self.labels.append(0)

        # Load deceptive videos (label 1)
        deceptive_dir = data_root / 'deceptive'
        if deceptive_dir.exists():
            for video_file in deceptive_dir.glob('*'):
                if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                    self.video_list.append(str(video_file))
                    self.labels.append(1)

        print(f"Loaded {len(self.video_list)} videos from {data_root}")
        print(f"Truthful: {self.labels.count(0)}, Deceptive: {self.labels.count(1)}")

    def _count_frames_manually(self, cap):
        """
        Manually count frames for videos with corrupted metadata
        """
        count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1
            if count > 10000:  # Safety limit
                break
        return count

    def _load_from_annotation(self, annotation_file):
        """Load videos from annotation file"""
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            if len(parts) >= 2:
                video_path = parts[0].strip()
                label = int(parts[1].strip())

                if os.path.exists(video_path):
                    self.video_list.append(video_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.video_list)} videos from {annotation_file}")
        print(f"Truthful: {self.labels.count(0)}, Deceptive: {self.labels.count(1)}")
    
    def _get_dummy_sample(self, label, video_path):
        """
        Return a safe dummy sample when video processing fails completely
        """
        return {
            'vision_behaviour': torch.zeros(self.num_frames, 50, dtype=torch.float32),
            'vision_face': torch.zeros(3, self.num_frames, self.frame_size[0], self.frame_size[1], dtype=torch.float32),
            'audio_mel': torch.zeros(3, self.n_mels, self.audio_length // 160 + 1, dtype=torch.float32),
            'audio_wave': torch.zeros(self.audio_length, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'videoname': os.path.basename(video_path) + '_DUMMY'
        }

    def _extract_audio(self, video_path):
        """
        Extract audio from video - WORKS WITH ALL FORMATS
        """
        try:
            import torchaudio
            import subprocess
            import tempfile
            
            # Try direct loading first
            try:
                waveform, sample_rate = torchaudio.load(video_path)
            except:
                # Fallback: extract audio to temp wav file using ffmpeg
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_path = temp_audio.name
                
                try:
                    # Extract audio using ffmpeg (handles all video formats)
                    subprocess.run([
                        'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                        '-ar', str(self.sample_rate), '-ac', '1', '-y', temp_path
                    ], check=True, capture_output=True)
                    
                    waveform, sample_rate = torchaudio.load(temp_path)
                    os.remove(temp_path)
                except Exception as e:
                    print(f"⚠️ Audio extraction failed for {os.path.basename(video_path)}: {str(e)}")
                    # Return silent audio
                    waveform = torch.zeros(1, self.audio_length)
                    sample_rate = self.sample_rate
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Pad or truncate to target length
            if waveform.shape[1] < self.audio_length:
                waveform = torch.nn.functional.pad(waveform, (0, self.audio_length - waveform.shape[1]))
            else:
                waveform = waveform[:, :self.audio_length]
            
            # Generate mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=400,
                hop_length=160
            )
            mel_spec = mel_transform(waveform)
            
            # Convert to 3-channel format (for compatibility)
            mel_spec = mel_spec.repeat(3, 1, 1)
            
            return waveform.squeeze(0).numpy(), mel_spec.numpy()
        
        except Exception as e:
            print(f"❌ Audio extraction error for {os.path.basename(video_path)}: {str(e)}")
            # Return silent audio
            return (np.zeros(self.audio_length, dtype=np.float32),
                    np.zeros((3, self.n_mels, self.audio_length // 160 + 1), dtype=np.float32))

    def _sample_frames(self, video_path):
        """
        Sample frames uniformly from video with FFmpeg fallback for problematic formats
        """
        video_name = os.path.basename(video_path)
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"❌ File does not exist: {video_path}")
            return None
        
        file_size = os.path.getsize(video_path)
        print(f"📹 Processing: {video_name} ({file_size/(1024*1024):.2f} MB)")
        
        # First attempt: OpenCV direct reading
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"⚠️ OpenCV failed to open {video_name}, trying FFmpeg fallback...")
            self.cap_release_safe(cap)
            return self._sample_frames_ffmpeg(video_path)
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"   Frames: {total_frames}, FPS: {fps:.1f}, Size: {width}x{height}")
            
            # Handle invalid metadata
            if total_frames <= 0 or fps <= 0:
                print(f"   ⚠️ Invalid metadata, trying FFmpeg fallback...")
                cap.release()
                return self._sample_frames_ffmpeg(video_path)
            
            # Test reading first frame
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print(f"   ❌ Cannot read frames, trying FFmpeg fallback...")
                cap.release()
                return self._sample_frames_ffmpeg(video_path)
            
            print(f"   ✅ First frame OK: {test_frame.shape}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset
            
            # Determine sampling indices
            if total_frames < self.num_frames:
                print(f"   ⚠️ Only {total_frames} frames, need {self.num_frames} (will duplicate)")
                indices = np.linspace(0, max(0, total_frames - 1), self.num_frames).astype(int)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            
            frames = []
            last_valid_frame = None
            failed_reads = 0
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    # Resize and convert
                    frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    last_valid_frame = frame.copy()
                else:
                    failed_reads += 1
                    # Use last valid frame or black frame
                    if last_valid_frame is not None:
                        frames.append(last_valid_frame.copy())
                    else:
                        frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))
            
            cap.release()
            
            if failed_reads > 0:
                print(f"   ⚠️ Failed to read {failed_reads}/{len(indices)} frames (used fallback)")
            
            # If too many failures, try FFmpeg
            if failed_reads > len(indices) // 2:
                print(f"   ❌ Too many failed reads ({failed_reads}/{len(indices)}), trying FFmpeg...")
                return self._sample_frames_ffmpeg(video_path)
            
            # Validate frame count
            if len(frames) != self.num_frames:
                while len(frames) < self.num_frames:
                    frames.append(frames[-1].copy() if frames else 
                                np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))
                frames = frames[:self.num_frames]
            
            result = np.array(frames, dtype=np.uint8)
            print(f"   ✅ Loaded {self.num_frames} frames, shape: {result.shape}")
            
            return result
        
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            cap.release()
            # Try FFmpeg as last resort
            print(f"   Trying FFmpeg fallback...")
            return self._sample_frames_ffmpeg(video_path)

    def cap_release_safe(self, cap):
        """Safely release VideoCapture"""
        try:
            if cap is not None:
                cap.release()
        except:
            pass
    
    def _sample_frames_ffmpeg(self, video_path):
        """
        Sample frames using FFmpeg (works with ALL video formats: mp4, mkv, wmv, avi, etc.)
        This is more robust than OpenCV for problematic codecs
        """
        video_name = os.path.basename(video_path)
        print(f"   🔧 Using FFmpeg for: {video_name}")

        # Check if FFmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=2, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(f"      ❌ FFmpeg not available, cannot process this video")
            return None
        
        # Initialize default values
        duration = 0.0
        fps = 30.0
        total_frames = self.num_frames * 2        

        try:
            # Get video duration and frame count using ffprobe
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets,duration,r_frame_rate',
                '-of', 'csv=p=0',
                video_path
            ]
            
            try:
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 2:
                        duration = float(parts[0]) if parts[0] else 0
                        fps_str = parts[1] if len(parts) > 1 else "30/1"
                        
                        # Parse FPS (format: "30/1" or "29.97")
                        if '/' in fps_str:
                            num, den = map(float, fps_str.split('/'))
                            fps = num / den if den > 0 else 30.0
                        else:
                            fps = float(fps_str) if fps_str else 30.0
                        
                        total_frames = int(duration * fps) if duration > 0 else self.num_frames * 2
                        print(f"      Video info: {duration:.1f}s, {fps:.1f} FPS, ~{total_frames} frames")
                    else:
                        total_frames = self.num_frames * 2
                else:
                    total_frames = self.num_frames * 2
            except:
                total_frames = self.num_frames * 2
            
            # Calculate timestamps to extract
            if total_frames < self.num_frames:
                timestamps = np.linspace(0, max(0.1, duration - 0.1), self.num_frames)
            else:
                timestamps = np.linspace(0, duration - 0.1, self.num_frames) if duration > 0 else np.arange(self.num_frames) * 0.1
            
            frames = []
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Extract frames at specific timestamps
                for i, ts in enumerate(timestamps):
                    output_file = os.path.join(temp_dir, f'frame_{i:04d}.jpg')
                    
                    extract_cmd = [
                        'ffmpeg', '-ss', str(ts), '-i', video_path,
                        '-vframes', '1', '-vf', f'scale={self.frame_size[1]}:{self.frame_size[0]}',
                        '-y', output_file
                    ]
                    
                    try:
                        subprocess.run(extract_cmd, capture_output=True, timeout=5, check=False)
                        
                        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                            frame = cv2.imread(output_file)
                            if frame is not None:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frames.append(frame)
                            else:
                                # Use last valid or black frame
                                if frames:
                                    frames.append(frames[-1].copy())
                                else:
                                    frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))
                        else:
                            # Fallback
                            if frames:
                                frames.append(frames[-1].copy())
                            else:
                                frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))
                    
                    except subprocess.TimeoutExpired:
                        print(f"      ⚠️ Timeout extracting frame {i}")
                        if frames:
                            frames.append(frames[-1].copy())
                        else:
                            frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))
            
            finally:
                # Cleanup temp files
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
            
            # Ensure correct number of frames
            if len(frames) == 0:
                print(f"      ❌ FFmpeg failed to extract any frames")
                return np.zeros((self.num_frames, self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
                # return None
            
            while len(frames) < self.num_frames:
                frames.append(frames[-1].copy())
            
            frames = frames[:self.num_frames]
            
            result = np.array(frames, dtype=np.uint8)
            print(f"      ✅ FFmpeg extracted {len(frames)} frames, shape: {result.shape}")
            
            return result
        
        except Exception as e:
            print(f"      ❌ FFmpeg fallback failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_behavioral_features(self, frames):
        """
        Extract 50 behavioral features per frame:
        - 35 Action Units (AU)
        - 8 Gaze features
        - 5 Expression features
        - 2 Valence/Arousal features
        
        Returns: (num_frames, 50) numpy array
        """
        # Initialize MediaPipe in the worker process
        self._init_mediapipe()

        all_features = []
        faces_detected = 0

        for frame_idx in range(len(frames)):
            frame = frames[frame_idx]

            # Ensure correct format and dimensions
            if frame.shape[0] != self.frame_size[0] or frame.shape[1] != self.frame_size[1]:
                frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))

            # Ensure uint8 format for MediaPipe
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # Process with MediaPipe
            results = self.face_mesh.process(frame)

            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self._compute_features_from_landmarks(landmarks, frame.shape)
                faces_detected += 1
            else:
                # No face detected, use zero features
                features = np.zeros(50, dtype=np.float32)

            # Validate: Ensure exactly 50 features
            if features.shape != (50,):
                print(f"⚠️ WARNING: Feature shape is {features.shape}, expected (50,)")
                if features.shape[0] < 50:
                    features = np.pad(features, (0, 50 - features.shape[0]), mode='constant')
                else:
                    features = features[:50]
            
            all_features.append(features)
        
        # Report detection rate occasionally
        if random.random() < 0.05:  # 5% of videos
            print(f"  Face detection: {faces_detected}/{len(frames)} frames")

        return np.stack(all_features, axis=0)

    def _compute_features_from_landmarks(self, landmarks, image_shape):
        """
        Compute 50-dimensional feature vector from MediaPipe 478 face landmarks:
        
        Feature breakdown:
        [0:35]   - Action Units (35)
        [35:43]  - Gaze features (8)
        [43:48]  - Expression features (5)
        [48:50]  - Valence/Arousal (2)
        """
        h, w = image_shape[:2]
        
        # Convert landmarks to numpy array (x, y, z coordinates normalized)
        points = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks])
        
        features = []
        
        # ============================================================
        # PART 1: ACTION UNITS (35 features)
        # ============================================================
        
        # --- Eyes (12 features) ---
        # AU5 (Upper Lid Raiser), AU7 (Lid Tightener), AU43 (Eyes Closed)
        left_eye_openness = self._eye_aspect_ratio(points, side='left')
        right_eye_openness = self._eye_aspect_ratio(points, side='right')
        
        # Eye widths
        left_eye_width = np.linalg.norm(points[33] - points[133])
        right_eye_width = np.linalg.norm(points[362] - points[263])
        
        # Inter-eye distance
        eye_distance = np.linalg.norm(points[33] - points[263])
        
        # Upper/lower lid positions
        left_upper_lid = np.linalg.norm(points[159] - points[145])
        right_upper_lid = np.linalg.norm(points[386] - points[374])
        left_lower_lid = np.linalg.norm(points[145] - points[153])
        right_lower_lid = np.linalg.norm(points[374] - points[380])
        
        # Eye squint/tightening
        left_eye_squint = np.linalg.norm(points[159] - points[153])
        right_eye_squint = np.linalg.norm(points[386] - points[380])
        
        # Eye symmetry
        eye_symmetry = abs(left_eye_width - right_eye_width) / (eye_distance + 1e-6)
        
        features.extend([
            left_eye_openness, right_eye_openness, left_eye_width, right_eye_width,
            eye_distance, left_upper_lid, right_upper_lid, left_lower_lid, right_lower_lid,
            left_eye_squint, right_eye_squint, eye_symmetry
        ])  # 12 features
        
        # --- Eyebrows (8 features) ---
        # AU1 (Inner Brow Raiser), AU2 (Outer Brow Raiser), AU4 (Brow Lowerer)
        left_brow_height = self._eyebrow_height(points, side='left')
        right_brow_height = self._eyebrow_height(points, side='right')
        
        # Inner brow points
        left_inner_brow = np.linalg.norm(points[70] - points[27])   # Left inner brow to nose bridge
        right_inner_brow = np.linalg.norm(points[300] - points[27]) # Right inner brow to nose bridge
        
        # Outer brow points
        left_outer_brow = np.linalg.norm(points[105] - points[33])  # Left outer brow to eye
        right_outer_brow = np.linalg.norm(points[334] - points[263]) # Right outer brow to eye
        
        # Brow distance and angle
        brow_distance = np.linalg.norm(points[70] - points[300])
        brow_angle = np.arctan2(points[300][1] - points[70][1], 
                                points[300][0] - points[70][0])
        
        features.extend([
            left_brow_height, right_brow_height, left_inner_brow, right_inner_brow,
            left_outer_brow, right_outer_brow, brow_distance, brow_angle
        ])  # 8 features
        
        # --- Mouth (10 features) ---
        # AU10 (Upper Lip Raiser), AU12 (Lip Corner Puller/Smile), AU15 (Lip Corner Depressor)
        # AU20 (Lip Stretcher), AU23 (Lip Tightener), AU25 (Lips Part), AU26 (Jaw Drop)
        mouth_aspect_ratio = self._mouth_aspect_ratio(points)
        
        # Mouth dimensions
        mouth_width = np.linalg.norm(points[61] - points[291])
        mouth_height = np.linalg.norm(points[13] - points[14])
        
        # Lip positions
        upper_lip_center = np.linalg.norm(points[0] - points[13])
        lower_lip_center = np.linalg.norm(points[17] - points[14])
        
        # Lip corners
        left_corner_height = np.linalg.norm(points[61] - points[291]) 
        right_corner_height = np.linalg.norm(points[291] - points[61])
        
        # Mouth opening and lip distance
        mouth_opening = np.linalg.norm(points[13] - points[14])
        lip_distance = np.linalg.norm(points[0] - points[17])
        
        # Mouth asymmetry
        left_mouth = np.linalg.norm(points[61] - points[0])
        right_mouth = np.linalg.norm(points[291] - points[0])
        mouth_asymmetry = abs(left_mouth - right_mouth) / (mouth_width + 1e-6)
        
        features.extend([
            mouth_aspect_ratio, mouth_width, mouth_height, upper_lip_center, lower_lip_center,
            left_corner_height, right_corner_height, mouth_opening, lip_distance, mouth_asymmetry
        ])  # 10 features
        
        # --- Nose & Cheeks (5 features) ---
        # AU9 (Nose Wrinkler), AU11 (Nasolabial Deepener)
        nose_width = np.linalg.norm(points[129] - points[358])
        nose_tip_height = np.linalg.norm(points[1] - points[2])
        
        # Nasolabial folds (cheek to mouth)
        left_nasolabial = np.linalg.norm(points[206] - points[61])
        right_nasolabial = np.linalg.norm(points[426] - points[291])
        
        # Nose to chin
        nose_to_chin = np.linalg.norm(points[1] - points[152])
        
        features.extend([
            nose_width, nose_tip_height, left_nasolabial, right_nasolabial, nose_to_chin
        ])  # 5 features
        
        # Total AU features: 12 + 8 + 10 + 5 = 35 ✓
        
        # ============================================================
        # PART 2: GAZE FEATURES (8 features)
        # ============================================================
        
        # Head pose (pitch, yaw, roll)
        pitch, yaw, roll = self._head_pose(points, w, h)
        
        # Eye gaze direction (simplified estimation)
        left_eye_center = (points[33] + points[133]) / 2
        right_eye_center = (points[362] + points[263]) / 2
        nose_bridge = points[168]
        
        # Horizontal and vertical gaze for each eye
        left_gaze_h = (left_eye_center[0] - nose_bridge[0]) / w
        left_gaze_v = (left_eye_center[1] - nose_bridge[1]) / h
        right_gaze_h = (right_eye_center[0] - nose_bridge[0]) / w
        right_gaze_v = (right_eye_center[1] - nose_bridge[1]) / h
        
        # Gaze convergence (measure of focus)
        gaze_convergence = abs(left_gaze_h - right_gaze_h)
        
        features.extend([
            pitch, yaw, roll,
            left_gaze_h, left_gaze_v, right_gaze_h, right_gaze_v,
            gaze_convergence
        ])  # 8 features
        
        # ============================================================
        # PART 3: EXPRESSION FEATURES (5 features)
        # ============================================================
        
        # Overall facial symmetry
        symmetry = self._facial_symmetry(points)
        
        # Face dimensions
        face_width = np.linalg.norm(points[234] - points[454])
        face_height = np.linalg.norm(points[10] - points[152])
        
        # Expression activity indicators
        upper_face_activity = (left_brow_height + right_brow_height) / 2
        lower_face_activity = mouth_aspect_ratio
        
        features.extend([
            symmetry, face_width, face_height, upper_face_activity, lower_face_activity
        ])  # 5 features
        
        # ============================================================
        # PART 4: VALENCE/AROUSAL (2 features)
        # ============================================================
        
        # Valence: positive (smile) vs negative (frown)
        # Use mouth corner positions relative to face center
        mouth_corners_avg = (left_mouth + right_mouth) / 2
        valence = mouth_corners_avg / (face_height + 1e-6)
        
        # Arousal: high (alert, wide eyes) vs low (calm, relaxed)
        # Combine eye openness and mouth opening
        arousal = (left_eye_openness + right_eye_openness + mouth_aspect_ratio) / 3.0
        
        features.extend([valence, arousal])  # 2 features
        
        # ============================================================
        # FINALIZE
        # ============================================================
        
        features = np.array(features, dtype=np.float32)
        
        # Sanity check
        assert len(features) == 50, f"Expected 50 features, got {len(features)}"
        
        # Clip extreme values
        features = np.clip(features, -100, 100)
        
        # Normalize
        mean = features.mean()
        std = features.std()
        if std > 1e-6:
            features = (features - mean) / std
        
        return features

    def _eye_aspect_ratio(self, points, side='left'):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        if side == 'left':
            # Left eye landmarks
            p1, p2, p3, p4, p5, p6 = 33, 160, 158, 133, 153, 144
        else:
            # Right eye landmarks
            p1, p2, p3, p4, p5, p6 = 362, 385, 387, 263, 373, 380
        
        # Vertical distances
        v1 = np.linalg.norm(points[p2] - points[p6])
        v2 = np.linalg.norm(points[p3] - points[p5])
        
        # Horizontal distance
        h = np.linalg.norm(points[p1] - points[p4])
        
        # EAR formula
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear

    def _eyebrow_height(self, points, side='left'):
        """Calculate eyebrow height relative to eye"""
        if side == 'left':
            brow_point = points[70]   # Left eyebrow center
            eye_point = points[33]    # Left eye inner corner
        else:
            brow_point = points[300]  # Right eyebrow center
            eye_point = points[263]   # Right eye inner corner
        
        height = np.linalg.norm(brow_point - eye_point)
        return height

    def _mouth_aspect_ratio(self, points):
        """Calculate Mouth Aspect Ratio (MAR)"""
        # Upper and lower lip center points
        upper = points[13]
        lower = points[14]
        
        # Left and right mouth corners
        left = points[61]
        right = points[291]
        
        # Vertical distance
        v = np.linalg.norm(upper - lower)
        
        # Horizontal distance
        h = np.linalg.norm(left - right)
        
        # MAR formula
        mar = v / (h + 1e-6)
        return mar
    
    def _head_pose(self, points, w, h):
        """Estimate head pose (pitch, yaw, roll) from facial landmarks"""
        # Key points for pose estimation
        nose_tip = points[1]
        chin = points[152]
        left_eye = points[33]
        right_eye = points[263]
        left_mouth = points[61]
        right_mouth = points[291]
        
        # Yaw (left-right head rotation)
        eye_center = (left_eye + right_eye) / 2
        yaw = np.arctan2(nose_tip[0] - eye_center[0], nose_tip[2] - eye_center[2] + 1e-6)
        
        # Pitch (up-down head rotation)
        pitch = np.arctan2(nose_tip[1] - chin[1], abs(nose_tip[2] - chin[2]) + 1e-6)
        
        # Roll (head tilt)
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0] + 1e-6)
        
        return pitch, yaw, roll

    def _facial_symmetry(self, points):
        """Calculate facial symmetry score"""
        # Compare left and right landmarks
        left_landmarks = [33, 133, 61, 206]  # Left eye, mouth, cheek
        right_landmarks = [263, 362, 291, 426]  # Right eye, mouth, cheek
        
        # Get face center (nose tip)
        center_x = points[1][0]
        
        # Calculate symmetry for each pair
        symmetry_scores = []
        for left_idx, right_idx in zip(left_landmarks, right_landmarks):
            left_dist = abs(points[left_idx][0] - center_x)
            right_dist = abs(points[right_idx][0] - center_x)
            
            # Symmetry score (closer to 1 = more symmetric)
            score = 1.0 - abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)
            symmetry_scores.append(score)
        
        return np.mean(symmetry_scores)

    def __len__(self):
        return len(self.video_list)

    # def __getitem__(self, idx):
    #     video_path = self.video_list[idx]
    #     label = self.labels[idx]

    #     # Enable verbose debugging for first 2 videos
    #     if idx < 2:
    #         print(f"\n\n{'#'*70}")
    #         print(f"# PROCESSING SAMPLE {idx}")
    #         print(f"# Video: {os.path.basename(video_path)}")
    #         print(f"{'#'*70}\n")
        
    #     # Extract frames with debugging
       

    #     try:
    #         frames = self._sample_frames(video_path)
        
    #         if frames is None  or frames.size == 0:
    #             print(f"❌ Frames is None, creating dummy frames")
    #             frames = np.zeros((self.num_frames, self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            
    #         # Extract behavioral features with debugging
    #         behavioral_features = self._extract_behavioral_features(frames)

            
    #         # Extract face frames
    #         if frames is None or frames.size == 0:
    #             frames = np.zeros((self.num_frames, self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)

    #         # Extract audio
    #         audio_wave, audio_mel = self._extract_audio(video_path)

    #         # Extract behavioral features (OpenFace + Affect)
    #         # behavioral_features = self._extract_behavioral_features(frames)

    #         # 🔍 Validate behavioral features shape
    #         if behavioral_features.shape != (self.num_frames, 64): #50
    #             print(f"⚠️ Invalid behavioral features shape {behavioral_features.shape} for {os.path.basename(video_path)}")
    #             behavioral_features = np.zeros((self.num_frames, 64), dtype=np.float32) #50
            
    #         # Convert face frames to tensor: (T, H, W, C) -> (C, T, H, W) for the model
    #         frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    #         # Normalize to [-1, 1]
    #         frames = (frames / 255.0 - 0.5) * 2.0

    #         # 🔍 DEBUG: Print shapes
    #         if idx % 50 == 0:  # Print for first sample
    #             print(f"\n🔍 Info for video: {os.path.basename(video_path)}")
    #             print(f"  Frames shape: {frames.shape}")
    #             print(f"  Behavioral features shape: {behavioral_features.shape}")
    #             print(f"  Behavioral features sample: {behavioral_features[0][:10]}")  # First 10 features
    #             print(f"  Non-zero features: {np.count_nonzero(behavioral_features)}/{behavioral_features.size}")
            
    #         # Convert behavioral features to tensor
    #         behavioral_features = torch.from_numpy(behavioral_features).float()
            
    #         sample = {
    #             'vision_behaviour': behavioral_features,  # (T=64, 64)
    #             'vision_face': frames,  # (C=3, T=64, H=160, W=160)
    #             'audio_mel': audio_mel,  # (C=3, n_mels=128, time)
    #             'audio_wave': audio_wave,  # (audio_length,)
    #             'label': torch.tensor(label, dtype=torch.long),
    #             'videoname': os.path.basename(video_path)
    #         }

    #         return sample

    #     except Exception as e:
    #         print(f"❌ Error processing {os.path.basename(video_path)}: {str(e)}")
    #         # Return a safe dummy sample
    #         return self._get_dummy_sample(label, video_path)

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        label = self.labels[idx]

        # Enable verbose debugging for first 2 videos
        if idx < 2:
            print(f"\n{'#'*60}")
            print(f"# PROCESSING SAMPLE {idx}: {os.path.basename(video_path)}")
            print(f"{'#'*60}")

        try:
            # LOAD FRAMES
            frames = self._sample_frames(video_path)
        
            if frames is None or frames.size == 0:
                print(f"❌ Frames is None, creating dummy frames")
                # Return dummy immediately, don't proceed
                return self._get_dummy_sample(label, video_path)
            
            # EXTRACT BEHAVIORAL FEATURES
            # This returns shape (64, 50)
            behavioral_features = self._extract_behavioral_features(frames)

            # FIX THE SHAPE MISMATCH (The Fix for your Warning)
            EXPECTED_RAW_DIM = 50
            
            # Validate shape
            if behavioral_features.shape[1] != EXPECTED_RAW_DIM:
                print(f"⚠️ Fixing behavioral shape: {behavioral_features.shape} -> {EXPECTED_RAW_DIM}")
                
                if behavioral_features.shape[1] < EXPECTED_RAW_DIM:
                    # Pad if we somehow got less than 50 (rare error case)
                    padding = np.zeros((self.num_frames, EXPECTED_RAW_DIM - behavioral_features.shape[1]), dtype=np.float32)
                    behavioral_features = np.concatenate([behavioral_features, padding], axis=1)
                else:
                    # Crop if we got more
                    behavioral_features = behavioral_features[:, :EXPECTED_RAW_DIM]

            # 4. EXTRACT AUDIO
            audio_wave, audio_mel = self._extract_audio(video_path)

            # 5. PREPARE TENSORS
            # Convert face frames to tensor: (T, H, W, C) -> (C, T, H, W)
            frames_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
            # Normalize to [-1, 1]
            frames_tensor = (frames_tensor / 255.0 - 0.5) * 2.0

            # Convert behavioral features to tensor
            behavioral_tensor = torch.from_numpy(behavioral_features).float()
            
            # 6. DEBUG (Optional)
            if idx % 50 == 0:
                print(f"✅ Loaded {os.path.basename(video_path)} | Feat Shape: {behavioral_tensor.shape}")

            sample = {
                'vision_behaviour': behavioral_tensor,  # (T=64, 64)
                'vision_face': frames_tensor,           # (C=3, T=64, H=160, W=160)
                'audio_mel': audio_mel,                 # (C=3, n_mels=128, time)
                'audio_wave': audio_wave,               # (audio_length,)
                'label': torch.tensor(label, dtype=torch.long),
                'videoname': os.path.basename(video_path)
            }

            return sample

        except Exception as e:
            print(f"❌ Error processing {os.path.basename(video_path)}: {str(e)}")
            return self._get_dummy_sample(label, video_path)

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


        # BACKWARD PASS WITH SCALER
        # Scales loss to prevent underflow in float16
        scaler.scale(loss).backward()

        # Gradient clipping (Unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()

        # # Backward pass
        # loss.backward()

       
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

# def validate(model, dataloader, criterion, device):
#     """Validate the model"""
#     model.eval()

#     loss_meter = AvgrageMeter()
#     all_preds = []
#     all_labels = []
#     all_scores = []

#     with torch.no_grad():
#         for sample_batched in dataloader:
#             vision_behaviour = sample_batched['vision_behaviour'].to(device)
#             vision_face = sample_batched['vision_face'].to(device)
#             audio_mel = sample_batched['audio_mel'].to(device)
#             audio_wave = sample_batched['audio_wave'].to(device)
#             labels = sample_batched['label'].to(device)

#             # Forward pass
#             fused_logit, _, _, _, _ = model(
#                 vision_behaviour, vision_face, audio_mel, audio_wave
#             )

#             loss = criterion(fused_logit, labels)
#             loss_meter.update(loss.item(), vision_face.size(0))

#             probs = F.softmax(fused_logit, dim=1)
#             _, predicted = torch.max(fused_logit.data, 1)

#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_scores.extend(probs[:, 1].cpu().numpy())

#     # Calculate metrics
#     correct = sum([p == l for p, l in zip(all_preds, all_labels)])
#     accuracy = 100 * correct / len(all_labels)

#     return loss_meter.avg, accuracy, all_preds, all_labels, all_scores

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

def save_checkpoint(model, optimizer, epoch, current_step, best_val_acc, scheduler_warmup, scheduler_cosine, warmup_steps):
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
    }
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'latest.pt'))
    print(f"✅ Checkpoint saved: epoch {epoch + 1}")

def load_checkpoint(model, optimizer, scheduler_warmup, scheduler_cosine, device):
    """Load checkpoint if exists (after spot interruption)."""
    path = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler_warmup.load_state_dict(ckpt['scheduler_warmup_state'])
        scheduler_cosine.load_state_dict(ckpt['scheduler_cosine_state'])
        print(f"🔄 Resumed from epoch {ckpt['epoch'] + 1}")
        return ckpt['epoch'] + 1, ckpt['current_step'], ckpt['best_val_acc']
    return 0, 0, 0.0

def compute_class_weights(dataset):
    """
    Fast version - directly access labels without loading full samples
    Only use if your dataset has a direct label access method
    """
    print(f"\n{'='*60}")
    print(f"📊 Computing Class Weights (Fast Mode)...")
    print(f"{'='*60}")
    
    # Check if dataset has direct label access
    if hasattr(dataset, 'labels'):
        labels = dataset.labels
        print(f"  ✅ Using pre-loaded labels from dataset")
    else:
        print(f"  ⚠️ No direct label access, falling back to full loading")
        return compute_class_weights(dataset)
    
    class_counts = np.bincount(labels)
    total = sum(class_counts)
    num_classes = len(class_counts)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Class 0 (Truth): {class_counts[0]} samples ({100*class_counts[0]/total:.1f}%)")
    print(f"  Class 1 (Lie):   {class_counts[1]} samples ({100*class_counts[1]/total:.1f}%)")
    print(f"  Imbalance ratio: {max(class_counts)/min(class_counts):.2f}:1")
    
    # Compute inverse frequency weights
    weights = total / (num_classes * class_counts)
    weights = weights / weights.sum() * num_classes
    
    print(f"  Class weights: [Truth: {weights[0]:.3f}, Lie: {weights[1]:.3f}]")
    print(f"{'='*60}\n")
    
    return torch.FloatTensor(weights)


def main(args):

    # install_ffmpeg()

    # === START OF main(args) ===
    print(f"DEBUG: Checking path {args.train_root}")
    if os.path.exists(args.train_root):
        contents = os.listdir(args.train_root)
        print(f"DEBUG: Folder contents: {contents}")
        
        # Check for class folders
        for cls in ['truthful', 'deceptive']:
            cls_path = os.path.join(args.train_root, cls)
            if os.path.exists(cls_path):
                files = os.listdir(cls_path)
                print(f"DEBUG: Found '{cls}' with {len(files)} files.")
            else:
                print(f"DEBUG: ❌ CRITICAL: Folder '{cls}' MISSING in {args.train_root}")
    else:
        print(f"DEBUG: ❌ CRITICAL: Path {args.train_root} does not exist.")
    # ====================================

    # Setup
    setup_seed(42)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Create log directory
    # os.makedirs(args.log, exist_ok=True)
    # log_file = open(os.path.join(args.log, 'training_log.txt'), 'a')  # Changed to 'a' for resume
    log_file = open(os.path.join(LOG_DIR, 'training_log.txt'), 'a', buffering=1)
    


    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    log_file.write(f"Arguments: {args}\n")

    # Create datasets
    if args.train_root:
        train_dataset = VideoDeceptionDataset(
            data_root=args.train_root,
            num_frames=args.num_frames,
            frame_size=(args.frame_height, args.frame_width),
            audio_length=args.audio_length,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            mode='train'
        )
    else:
        train_dataset = VideoDeceptionDataset(
            annotation_file=args.train_list,
            num_frames=args.num_frames,
            frame_size=(args.frame_height, args.frame_width),
            audio_length=args.audio_length,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            mode='train'
        )

    if args.val_root:
        val_dataset = VideoDeceptionDataset(
            data_root=args.val_root,
            num_frames=args.num_frames,
            frame_size=(args.frame_height, args.frame_width),
            audio_length=args.audio_length,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            mode='val'
        )
    else:
        val_dataset = VideoDeceptionDataset(
            annotation_file=args.val_list,
            num_frames=args.num_frames,
            frame_size=(args.frame_height, args.frame_width),
            audio_length=args.audio_length,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            mode='val'
        )

    print("============= Class Weight ==========")
    class_weights = compute_class_weights(train_dataset)
    print(class_weights)
    # train_sampler = create_balanced_sampler(train_dataset)
   
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Add device to args for model
    args.device = device

    # Create model
    # model = LightweightFusionModule(args)
    model = MinimalFusionModule(args)
    # model = FusionModule(args)

    ###### FREEZING MODEL
    for param in model.audio_model.parameters():
        param.requires_grad = False
    for param in model.face_model.parameters():
        param.requires_grad = False

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"\n🚀 Detected {torch.cuda.device_count()} GPUs! Activating DataParallel.")
        model = nn.DataParallel(model)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    log_file.write(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")

    ### Focal loss
    # criterion = FocalLoss(
    #     alpha=class_weights.to(device),
    #     gamma=0.3  # Focusing parameter 2.0
    # )

    # Loss and optimizer
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

    # ⬇️ LOAD CHECKPOINT IF RESUMING
    start_epoch, current_step, best_val_acc = load_checkpoint(
        model, optimizer, scheduler_warmup, scheduler_cosine, device
    )

    scaler = GradScaler()

    # Training loop
    for epoch in range(start_epoch, args.max_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        print(f"{'=' * 50}")

        if epoch == 2:
            if isinstance(model, nn.DataParallel):
                actual_model = model.module
            else:
                actual_model = model
            print(" Unfreezing encoders...")
            for param in actual_model.audio_model.parameters():
                param.requires_grad = True
            for param in actual_model.face_model.parameters():
                param.requires_grad = True

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
        if (epoch + 1) % args.val_interval == 0:
            # val_loss, val_acc, val_preds, val_labels, val_scores = validate(
            #     model, val_loader, criterion, device
            # )
            val_loss, val_acc, val_f1, val_preds, val_labels, val_scores = validate(
                model, val_loader, criterion, device
            )

            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
            log_file.write(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}\n")
            # print(f"Epoch [{epoch + 1}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            # log_file.write(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                # Unwrap model before saving Best Model
                if isinstance(model, nn.DataParallel):
                    model_to_save = model.module
                else:
                    model_to_save = model
                
                filename = f'best_model_epoch_{epoch + 1}.pt'
                save_path = os.path.join(CHECKPOINT_DIR, filename)

                # Save to checkpoint dir (synced to S3)
                # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_val_acc,
                }, save_path)
                
                # # Also save to log dir
                # torch.save({
                #     'epoch': epoch + 1,
                #     'model_state_dict': model_to_save.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'best_acc': best_val_acc,
                # }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
                print(f"🏆 Saved best model with accuracy: {best_val_acc:.2f}%")
                log_file.write(f"Saved best model with accuracy: {best_val_acc:.2f}%\n")

        # Unwrap model before passing to custom save function
        if isinstance(model, nn.DataParallel):
            model_for_ckpt = model.module
        else:
            model_for_ckpt = model
        # ⬇️ SAVE CHECKPOINT AFTER EACH EPOCH
        save_checkpoint(
            model_for_ckpt, optimizer, epoch, current_step, best_val_acc,
            scheduler_warmup, scheduler_cosine, warmup_steps
        )

        log_file.flush()

    # Unwrap model before final save
    if isinstance(model, nn.DataParallel):
        final_model_to_save = model.module
    else:
        final_model_to_save = model

    # ⬇️ SAVE FINAL MODEL TO SAGEMAKER OUTPUT PATH
    # os.makedirs(MODEL_DIR, exist_ok=True)
    # torch.save(final_model_to_save.state_dict(), os.path.join(MODEL_DIR, 'model.pt'))
    
    torch.save({
        'model_state_dict': final_model_to_save.state_dict(),
        'args': vars(args),
        'best_acc': best_val_acc,
    }, os.path.join(MODEL_DIR, 'model_full.pt'))

    print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📦 Model saved to {MODEL_DIR}")
    log_file.write(f"\nBest validation accuracy: {best_val_acc:.2f}%\n")
    log_file.close()

if __name__ == "__main__":

    install_system_dependencies()

    try:
        # Force print immediately to prove code started
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
        parser.add_argument('--train_list', type=str, default=None,
                            help='Training annotation file')
        parser.add_argument('--val_list', type=str, default=None,
                            help='Validation annotation file')

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

        # Model parameters (from your original code)
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
        # This block catches ANY crash and prints it to your local terminal
        print("\n\n❌ ❌ CRITICAL FAILURE ❌ ❌", flush=True)
        print(str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        
        # # Optional: Write to failure file (SageMaker reads this)
        # with open('/opt/ml/output/failure', 'w') as f:
        #     f.write(str(e))
        #     f.write(traceback.format_exc())
        
        # Re-raise so the job is marked as Failed
        raise