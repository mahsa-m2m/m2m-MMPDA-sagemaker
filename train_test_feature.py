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




# SageMaker paths (with fallbacks for local testing)
CHECKPOINT_DIR = os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints')
MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

# ==================== COMPREHENSIVE WARNING SUPPRESSION ====================

# 1. Suppress Python warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# # 2. Suppress TensorFlow/MediaPipe logs (must be set BEFORE importing mediapipe)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only
# os.environ['GLOG_minloglevel'] = '3'  # Google logging
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# # 3. Suppress OpenCV/FFmpeg warnings
# os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
# os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
# os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '0'
# cv2.setLogLevel(0)  # 0 = silent


# 4. Redirect stderr temporarily for library initialization
# class SuppressOutput:
#     def __enter__(self):
#         self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
#         self.save_fds = [os.dup(1), os.dup(2)]
#         os.dup2(self.null_fds[0], 1)
#         os.dup2(self.null_fds[1], 2)
#         return self

#     def __exit__(self, *_):
#         os.dup2(self.save_fds[0], 1)
#         os.dup2(self.save_fds[1], 2)
#         for fd in self.null_fds + self.save_fds:
#             os.close(fd)


# # Import with suppression
# with SuppressOutput():
#     # 5. Suppress ABSL logging (MediaPipe uses this)
#     import absl.logging

#     absl.logging.set_verbosity(absl.logging.ERROR)
#     absl.logging.set_stderrthreshold(absl.logging.ERROR)

#     import mediapipe as mp

# Now import librosa with proper backend (without resampy dependency)
# FIX: Use scipy resampler instead of resampy
os.environ['LIBROSA_RESAMPLE_BACKEND'] = 'scipy'
import librosa

# ============================================================================

# Import your models
from models_comp.fusion_model import FusionModule
from utils import AvgrageMeter, performances
import DALoss
import DANetwork


# orig_stdout = sys.stdout
# orig_stderr = sys.stderr

# class Logger(object):
#     def __init__(self, filename, stream):
#         self.terminal = stream
#         self.log = open(filename, "a")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush()

#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()

# # 1. Define the checkpoint path
# chk_dir = os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints')
# os.makedirs(chk_dir, exist_ok=True)

# # 2. Redirect
# log_path = os.path.join(chk_dir, 'full_execution_log.txt')
# sys.stdout = Logger(log_path, orig_stdout)
# sys.stderr = Logger(log_path, orig_stderr)

# print(f"✅ Logging started to {log_path}")


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
                 num_frames=64, frame_size=(160, 160),
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
    
    def cap_release_safe(cap):
        """Safely release VideoCapture"""
        try:
            if cap is not None:
                cap.release()
        except:
            pass

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
            'vision_behaviour': torch.zeros(self.num_frames, 64, dtype=torch.float32),
            'vision_face': torch.zeros(3, self.num_frames, self.frame_size[0], self.frame_size[1], dtype=torch.float32),
            'audio_mel': torch.zeros(3, self.n_mels, self.audio_length // 160 + 1, dtype=torch.float32),
            'audio_wave': torch.zeros(self.audio_length, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'videoname': os.path.basename(video_path) + '_DUMMY'
        }
    
    # def _extract_audio(self, video_path):
        """
        Extract audio from video and create mel spectrogram
        FIX: Use scipy resampler (no resampy dependency) and torchaudio as fallback
        """
        try:
            # Method 1: Try librosa with scipy backend (no resampy needed)
            y, sr = librosa.load(
                video_path,
                sr=self.sample_rate,
                mono=True,
                res_type='soxr_hq'  # High-quality scipy resampler
            )

            # Convert to torch tensor
            waveform = torch.from_numpy(y).float()

        except Exception as e1:
            try:
                # Method 2: Fallback to torchaudio (works with more formats)
                waveform, sr = torchaudio.load(video_path)

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0)
                else:
                    waveform = waveform.squeeze(0)

                # Resample if needed using torchaudio
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

            except Exception as e2:
                # Method 3: Extract audio using FFmpeg first
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        tmp_path = tmp.name

                    cmd = [
                        'ffmpeg', '-i', video_path,
                        '-vn', '-acodec', 'pcm_s16le',
                        '-ar', str(self.sample_rate),
                        '-ac', '1', '-y',
                        '-loglevel', 'quiet',
                        tmp_path
                    ]
                    subprocess.run(cmd, check=True, timeout=30,
                                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

                    # Load extracted audio
                    y, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
                    waveform = torch.from_numpy(y).float()

                    os.remove(tmp_path)

                except Exception as e3:
                    # All methods failed - use silent audio
                    if random.random() < 0.05:  # Print 5% of errors
                        print(f"Note: All audio extraction methods failed for {os.path.basename(video_path)}")
                    waveform = torch.zeros(self.audio_length)

        # Pad or truncate to desired length
        if waveform.shape[0] < self.audio_length:
            padding = self.audio_length - waveform.shape[0]
            waveform = F.pad(waveform.unsqueeze(0), (0, padding)).squeeze(0)
        else:
            waveform = waveform[:self.audio_length]

        # Add channel dimension for mel transform
        waveform_with_channel = waveform.unsqueeze(0)

        # Create mel spectrogram with optimized parameters
        n_fft = 2048
        hop_length = 512

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=self.n_mels,
            f_min=20.0,
            f_max=min(8000.0, self.sample_rate // 2),
            power=2.0,
            norm='slaney',
            mel_scale='htk'
        )

        mel_spec = mel_transform(waveform_with_channel)

        # Convert to dB scale with proper reference
        mel_spec_db = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=80.0
        )(mel_spec)

        # Normalize to [-1, 1] range for better neural network training
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        mel_spec_db = torch.clamp(mel_spec_db, -3, 3)
        mel_spec_db = mel_spec_db / 3.0

        # Convert to 3-channel format for ResNet
        mel_spec_3ch = mel_spec_db.repeat(3, 1, 1)

        return waveform, mel_spec_3ch

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

    # def _sample_frames(self, video_path):
    #     """
    #     Sample frames uniformly from video with proper error handling
    #     FIX: Use FFmpeg fallback for videos that OpenCV can't read (especially MKV files)
    #     """
    #     # First attempt: OpenCV direct reading
    #     cap = cv2.VideoCapture(video_path)

    #     if not cap.isOpened():
    #         # Fallback: Use FFmpeg to pipe frames directly
    #         # return self._sample_frames_ffmpeg(video_path)
    #         return None

    #     # Set backend to FFmpeg for better H.264 handling
    #     cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    #     try:
    #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         fps = cap.get(cv2.CAP_PROP_FPS)

    #         # # FIX: If frame count is unreliable (common with H.264), estimate from duration
    #         # if total_frames <= 0 or fps <= 0:
    #         #     cap.release()
    #         #     return self._sample_frames_ffmpeg(video_path)

    #         # Handle videos with invalid metadata
    #         if total_frames <= 0 or fps <= 0:
    #             print(f"⚠️ Invalid video metadata for {os.path.basename(video_path)}, counting frames manually")
    #             total_frames = self._count_frames_manually(cap)
    #             cap.release()
    #             cap = cv2.VideoCapture(video_path)  # Reopen
            
    #         if total_frames < self.num_frames:
    #             print(f"⚠️ Video {os.path.basename(video_path)} has only {total_frames} frames, needed {self.num_frames}")
    #             # Sample with repetition
    #             indices = np.linspace(0, max(0, total_frames - 1), self.num_frames).astype(int)
    #         else:
    #             # Uniform sampling
    #             indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        
    #         frames = []
    #         last_valid_frame = None
    #         # failed_reads = 0
    #         # max_failures = self.num_frames // 4  # Allow up to 25% read failures

    #         for idx in indices:
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    #             ret, frame = cap.read()
                
    #             if ret and frame is not None:
    #                 # Resize to target size
    #                 frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
    #                 # Convert BGR to RGB
    #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 frames.append(frame)
    #                 last_valid_frame = frame.copy()
    #             else:
    #                 # Use last valid frame or create black frame
    #                 if last_valid_frame is not None:
    #                     frames.append(last_valid_frame.copy())
    #                 else:
    #                     black_frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
    #                     frames.append(black_frame)
    #         cap.release()

    #         if len(frames) != self.num_frames:
    #             print(f"⚠️ Expected {self.num_frames} frames, got {len(frames)}")
    #             # Pad or truncate
    #             while len(frames) < self.num_frames:
    #                 frames.append(frames[-1].copy() if frames else 
    #                             np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))
    #             frames = frames[:self.num_frames]
            
    #         return np.array(frames, dtype=np.uint8)
        
    #     except Exception as e:
    #         print(f"❌ Error sampling frames from {os.path.basename(video_path)}: {str(e)}")
    #         cap.release()
    #         return None

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
            cap_release_safe(cap)
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

    # def _sample_frames_ffmpeg(self, video_path):
    #     """
    #     Fallback method: Use FFmpeg directly to extract frames
    #     This handles MKV and other problematic formats that OpenCV can't read
    #     """
    #     try:
    #         # First, get video duration and fps using ffprobe
    #         probe_cmd = [
    #             'ffprobe',
    #             '-v', 'error',
    #             '-select_streams', 'v:0',
    #             '-show_entries', 'stream=duration,nb_frames,r_frame_rate',
    #             '-of', 'default=noprint_wrappers=1',
    #             video_path
    #         ]

    #         probe_result = subprocess.run(
    #             probe_cmd,
    #             capture_output=True,
    #             text=True,
    #             timeout=10
    #         )

    #         # Parse output
    #         duration = None
    #         nb_frames = None
    #         fps = None

    #         for line in probe_result.stdout.split('\n'):
    #             if 'duration=' in line:
    #                 try:
    #                     duration = float(line.split('=')[1])
    #                 except:
    #                     pass
    #             elif 'nb_frames=' in line:
    #                 try:
    #                     nb_frames = int(line.split('=')[1])
    #                 except:
    #                     pass
    #             elif 'r_frame_rate=' in line:
    #                 try:
    #                     rate_parts = line.split('=')[1].split('/')
    #                     fps = float(rate_parts[0]) / float(rate_parts[1])
    #                 except:
    #                     pass

    #         # Estimate total frames
    #         if nb_frames:
    #             total_frames = nb_frames
    #         elif duration and fps:
    #             total_frames = int(duration * fps)
    #         else:
    #             total_frames = self.num_frames  # Fallback

    #         # Calculate frame indices to extract
    #         if total_frames < self.num_frames:
    #             frame_indices = list(range(total_frames))
    #             # Duplicate last frame if needed
    #             while len(frame_indices) < self.num_frames:
    #                 frame_indices.append(frame_indices[-1] if frame_indices else 0)
    #         else:
    #             frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

    #         frames = []

    #         # Extract frames using FFmpeg
    #         for idx in frame_indices:
    #             # Seek to specific frame and extract one frame
    #             cmd = [
    #                 'ffmpeg',
    #                 '-ss', str(idx / max(fps, 1)),  # Seek to time position
    #                 '-i', video_path,
    #                 '-vframes', '1',  # Extract 1 frame
    #                 '-f', 'rawvideo',
    #                 '-pix_fmt', 'rgb24',
    #                 '-s', f'{self.frame_size[0]}x{self.frame_size[1]}',
    #                 '-v', 'quiet',
    #                 '-'
    #             ]

    #             result = subprocess.run(
    #                 cmd,
    #                 capture_output=True,
    #                 timeout=5
    #             )

    #             if result.returncode == 0 and len(result.stdout) > 0:
    #                 # Parse raw RGB data
    #                 expected_size = self.frame_size[0] * self.frame_size[1] * 3
    #                 if len(result.stdout) >= expected_size:
    #                     frame_data = np.frombuffer(result.stdout[:expected_size], dtype=np.uint8)
    #                     frame = frame_data.reshape((self.frame_size[1], self.frame_size[0], 3))
    #                     frames.append(frame)
    #                 else:
    #                     if frames:
    #                         frames.append(frames[-1].copy())
    #                     else:
    #                         frames.append(np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8))
    #             else:
    #                 if frames:
    #                     frames.append(frames[-1].copy())
    #                 else:
    #                     frames.append(np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8))

    #         # Ensure correct number of frames
    #         while len(frames) < self.num_frames:
    #             if frames:
    #                 frames.append(frames[-1].copy())
    #             else:
    #                 frames.append(np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8))

    #         frames = np.stack(frames[:self.num_frames], axis=0)
    #         return frames

    #     except Exception as e:
    #         if self.mode == 'train' and random.random() < 0.1:
    #             print(f"FFmpeg fallback also failed for {os.path.basename(video_path)}: {str(e)[:50]}")
    #         # Return black frames as last resort
    #         return np.zeros((self.num_frames, self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
    def _sample_frames_ffmpeg(self, video_path):
        """
        Sample frames using FFmpeg (works with ALL video formats: mp4, mkv, wmv, avi, etc.)
        This is more robust than OpenCV for problematic codecs
        """
        video_name = os.path.basename(video_path)
        print(f"   🔧 Using FFmpeg for: {video_name}")
        
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
                return None
            
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
        Extract behavioral features from MediaPipe face landmarks

        This extracts 64 features per frame including:
        - Head pose estimation (3 rotation angles: pitch, yaw, roll)
        - Eye aspect ratios (2 features: left, right eye openness)
        - Mouth aspect ratio (1 feature: mouth openness)
        - Eyebrow positions (2 features: left, right eyebrow height)
        - Facial symmetry (1 feature)
        - Key landmark distances (55 features: distances between important points)

        Total: 64 features per frame
        Expected output: (T, 64) where T=num_frames
        """
        # Initialize MediaPipe in the worker process
        self._init_mediapipe()

        all_features = []
        faces_detected = 0

        for frame_idx in range(len(frames)):
            frame = frames[frame_idx]

            # MediaPipe expects RGB images (frames are already in RGB from _sample_frames)
            # Ensure correct format and dimensions
            if frame.shape[0] != self.frame_size[0] or frame.shape[1] != self.frame_size[1]:
                frame = cv2.resize(frame, self.frame_size)

            # Ensure uint8 format for MediaPipe
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            ######## DEBUG: Check frame quality
            if frame_idx == 0 and random.random() < 0.01:  # 1% of videos
                print(f"  Frame quality: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
                print(f"  Frame is all black: {frame.max() == 0}")
        
            # Process with MediaPipe
            # FIX: Provide image dimensions to resolve NORM_RECT warning
            results = self.face_mesh.process(frame)

            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self._compute_features_from_landmarks(landmarks, frame.shape)
                faces_detected += 1
            else:
                # No face detected, use zero features
                features = np.zeros(64, dtype=np.float32)
                # features = np.random.randn(64).astype(np.float32) * 0.1

                    # 🔍 VALIDATE: Ensure exactly 64 features
            if features.shape != (64,):
                print(f"⚠️ WARNING: Feature shape is {features.shape}, padding/truncating to (64,)")
                if features.shape[0] < 64:
                    # Pad with zeros
                    features = np.pad(features, (0, 64 - features.shape[0]), mode='constant')
                else:
                    # Truncate
                    features = features[:64]
            
            all_features.append(features)
        
        ###### DEBUG: Report detection rate
        if random.random() < 0.05:  # 5% of videos
            print(f"  Face detection: {faces_detected}/{len(frames)} frames")

        return np.stack(all_features, axis=0)

    def _compute_features_from_landmarks(self, landmarks, frame_shape):
        """
        Compute 64 behavioral features from MediaPipe landmarks

        MediaPipe provides 478 3D landmarks. We extract meaningful features:
        """
        h, w = frame_shape[:2]

        # Convert landmarks to numpy array (normalized coordinates)
        lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        features = []

        # --- 1. Head Pose Estimation (3 features) ---
        # Using specific facial points for pose estimation
        nose_tip = lm_array[1]  # Nose tip
        chin = lm_array[152]  # Chin
        left_eye = lm_array[33]  # Left eye corner
        right_eye = lm_array[263]  # Right eye corner
        left_mouth = lm_array[61]  # Left mouth corner
        right_mouth = lm_array[291]  # Right mouth corner

        # Approximate pitch (up/down tilt)
        pitch = nose_tip[1] - chin[1]

        # Approximate yaw (left/right rotation)
        yaw = (right_eye[0] - left_eye[0]) - 0.5  # Normalized difference

        # Approximate roll (head tilt)
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

        features.extend([pitch, yaw, roll])

        # --- 2. Eye Aspect Ratios (2 features) ---
        # Left eye landmarks: 33, 160, 158, 133, 153, 144
        left_eye_points = lm_array[[33, 160, 158, 133, 153, 144]]
        left_ear = self._eye_aspect_ratio(left_eye_points)

        # Right eye landmarks: 263, 387, 385, 362, 380, 373
        right_eye_points = lm_array[[263, 387, 385, 362, 380, 373]]
        right_ear = self._eye_aspect_ratio(right_eye_points)

        features.extend([left_ear, right_ear])

        # --- 3. Mouth Aspect Ratio (1 feature) ---
        # Mouth landmarks: 61, 291, 0, 17, 84, 314
        mouth_points = lm_array[[61, 291, 0, 17, 84, 314]]
        mar = self._mouth_aspect_ratio(mouth_points)
        features.append(mar)

        # --- 4. Eyebrow Heights (2 features) ---
        # Left eyebrow: 70, Right eyebrow: 300
        left_eyebrow = lm_array[70][1]
        right_eyebrow = lm_array[300][1]
        features.extend([left_eyebrow, right_eyebrow])

        # --- 5. Facial Symmetry (1 feature) ---
        # Compare left and right sides
        left_side = lm_array[234]  # Left face boundary
        right_side = lm_array[454]  # Right face boundary
        face_center = lm_array[1]  # Nose tip as center

        left_dist = np.linalg.norm(left_side - face_center)
        right_dist = np.linalg.norm(right_side - face_center)
        symmetry = abs(left_dist - right_dist)
        features.append(symmetry)

        # --- 6. Key Landmark Distances (55 features) ---
        # Distances between important landmark pairs
        important_pairs = [
            (33, 263),  # Eye to eye
            (61, 291),  # Mouth corners
            (1, 152),  # Nose to chin
            (10, 152),  # Forehead to chin
            (33, 61),  # Left eye to left mouth
            (263, 291),  # Right eye to right mouth
            (33, 1),  # Left eye to nose
            (263, 1),  # Right eye to nose
            (61, 1),  # Left mouth to nose
            (291, 1),  # Right mouth to nose
            # Add more pairs for total 55 features
        ]

        # Calculate distances for important pairs
        for i, j in important_pairs:
            dist = np.linalg.norm(lm_array[i] - lm_array[j])
            features.append(dist)

        # Add more distance features to reach 55
        # Using contour points and other key landmarks
        additional_indices = [
            (70, 300),  # Eyebrow to eyebrow
            (33, 133),  # Left eye width
            (263, 362),  # Right eye width
            (78, 308),  # Upper lip
            (13, 14),  # Lower face
            (10, 152),  # Full face height
            (234, 454),  # Face width
            (127, 356),  # Nose width
        ]

        for i, j in additional_indices:
            dist = np.linalg.norm(lm_array[i] - lm_array[j])
            features.append(dist)

        # Fill remaining features with landmark position statistics
        while len(features) < 64:
            # Add variance and mean of landmark positions as features
            if len(features) < 64:
                features.append(np.mean(lm_array[:, 0]))  # Mean x
            if len(features) < 64:
                features.append(np.mean(lm_array[:, 1]))  # Mean y
            if len(features) < 64:
                features.append(np.std(lm_array[:, 0]))  # Std x
            if len(features) < 64:
                features.append(np.std(lm_array[:, 1]))  # Std y
            if len(features) < 64:
                features.append(np.mean(lm_array[:, 2]))  # Mean z (depth)
            if len(features) < 64:
                features.append(np.std(lm_array[:, 2]))  # Std z
            if len(features) < 64:
                # Add more statistical features
                features.append(np.max(lm_array[:, 0]) - np.min(lm_array[:, 0]))  # x range
            if len(features) < 64:
                features.append(np.max(lm_array[:, 1]) - np.min(lm_array[:, 1]))  # y range

        # Ensure exactly 64 features
        features = np.array(features[:64], dtype=np.float32)

        # 🔍 ENSURE EXACTLY 64
        if len(features) < 64:
            # Pad with zeros
            features = np.pad(features, (0, 64 - len(features)), mode='constant')
        elif len(features) > 64:
            # Truncate
            features = features[:64]
        
        assert features.shape == (64,), f"❌ Features shape {features.shape}, expected (64,)"
        

        return features

    def _eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR) for eye openness"""
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])

        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])

        # EAR formula
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear

    def _mouth_aspect_ratio(self, mouth_points):
        """Calculate Mouth Aspect Ratio (MAR) for mouth openness"""
        # Vertical distances
        v1 = np.linalg.norm(mouth_points[2] - mouth_points[3])
        v2 = np.linalg.norm(mouth_points[4] - mouth_points[5])

        # Horizontal distance
        h = np.linalg.norm(mouth_points[0] - mouth_points[1])

        # MAR formula
        mar = (v1 + v2) / (2.0 * h + 1e-6)
        return mar

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        label = self.labels[idx]

        # Enable verbose debugging for first 2 videos
        if idx < 2:
            print(f"\n\n{'#'*70}")
            print(f"# PROCESSING SAMPLE {idx}")
            print(f"# Video: {os.path.basename(video_path)}")
            print(f"{'#'*70}\n")
        
        # Extract frames with debugging
        frames = self._sample_frames(video_path)
        
        if frames is None:
            print(f"❌ Frames is None, creating dummy frames")
            frames = np.zeros((self.num_frames, self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
        
        # Extract behavioral features with debugging
        behavioral_features = self._extract_behavioral_features(frames)

        try:
            # Extract face frames
            frames = self._sample_frames(video_path)
            if frames is None or frames.size == 0:
                frames = np.zeros((self.num_frames, self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)

            # Extract audio
            audio_wave, audio_mel = self._extract_audio(video_path)

            # Extract behavioral features (OpenFace + Affect)
            behavioral_features = self._extract_behavioral_features(frames)

            # 🔍 Validate behavioral features shape
            if behavioral_features.shape != (self.num_frames, 64):
                print(f"⚠️ Invalid behavioral features shape {behavioral_features.shape} for {os.path.basename(video_path)}")
                behavioral_features = np.zeros((self.num_frames, 64), dtype=np.float32)
            
            # Convert face frames to tensor: (T, H, W, C) -> (C, T, H, W) for the model
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
            # Normalize to [-1, 1]
            frames = (frames / 255.0 - 0.5) * 2.0

            # 🔍 DEBUG: Print shapes
            if idx == 0:  # Print for first sample
                print(f"\n🔍 Info for video: {os.path.basename(video_path)}")
                print(f"  Frames shape: {frames.shape}")
                print(f"  Behavioral features shape: {behavioral_features.shape}")
                print(f"  Behavioral features sample: {behavioral_features[0][:10]}")  # First 10 features
                print(f"  Non-zero features: {np.count_nonzero(behavioral_features)}/{behavioral_features.size}")
            
            # Convert behavioral features to tensor
            behavioral_features = torch.from_numpy(behavioral_features).float()

            # # 🔍 DEBUG: Print tensor shapes
            # if idx == 0:
            #     print(f"  Behavioral tensor shape: {behavioral_features.shape}")
            #     print(f"  Expected shape: ({self.num_frames}, 64)\n")
            
            sample = {
                'vision_behaviour': behavioral_features,  # (T=64, 64)
                'vision_face': frames,  # (C=3, T=64, H=160, W=160)
                'audio_mel': audio_mel,  # (C=3, n_mels=128, time)
                'audio_wave': audio_wave,  # (audio_length,)
                'label': torch.tensor(label, dtype=torch.long),
                'videoname': os.path.basename(video_path)
            }

            return sample

        except Exception as e:
            print(f"❌ Error processing {os.path.basename(video_path)}: {str(e)}")
            # Return a safe dummy sample
            return self._get_dummy_sample(label, video_path)

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """Train for one epoch"""
    model.train()

    loss_global = AvgrageMeter()
    loss_vl = AvgrageMeter()
    loss_face = AvgrageMeter()
    loss_al = AvgrageMeter()

    correct = 0
    total = 0

    for i, sample_batched in enumerate(dataloader):
        # Get data
        vision_behaviour = sample_batched['vision_behaviour'].to(device)  # (B, T, 64)
        vision_face = sample_batched['vision_face'].to(device)  # (B, C, T, H, W)
        audio_mel = sample_batched['audio_mel'].to(device)  # (B, 3, n_mels, time)
        audio_wave = sample_batched['audio_wave'].to(device)  # (B, audio_length)
        labels = sample_batched['label'].to(device)  # (B,)

        # Forward pass
        optimizer.zero_grad()

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
            loss = global_loss + vl_loss + face_loss + al_loss
        else:
            loss = global_loss
            vl_loss = torch.tensor(0.0)
            face_loss = torch.tensor(0.0)
            al_loss = torch.tensor(0.0)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

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
    """Validate the model"""
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

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    accuracy = 100 * correct / len(all_labels)

    return loss_meter.avg, accuracy, all_preds, all_labels, all_scores

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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
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

# def install_ffmpeg():
#     print("Installing ffmpeg...")
#     subprocess.run(["sudo apt-get", "update"], check=True)
#     subprocess.run(["sudo apt-get", "install", "-y", "ffmpeg"], check=True)  # ❌ Remove
    
def main(args):

    # install_ffmpeg()

    # === PASTE AT START OF main(args) ===
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
    os.makedirs(args.log, exist_ok=True)
    log_file = open(os.path.join(args.log, 'training_log.txt'), 'a')  # Changed to 'a' for resume

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
    model = FusionModule(args)
    model = model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    log_file.write(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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

    # Training loop
    for epoch in range(start_epoch, args.max_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        print(f"{'=' * 50}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, args
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
            val_loss, val_acc, val_preds, val_labels, val_scores = validate(
                model, val_loader, criterion, device
            )

            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            log_file.write(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # Save to checkpoint dir (synced to S3)
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_val_acc,
                }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
                
                # Also save to log dir
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_val_acc,
                }, os.path.join(args.log, 'best_model.pt'))
                print(f"🏆 Saved best model with accuracy: {best_val_acc:.2f}%")
                log_file.write(f"Saved best model with accuracy: {best_val_acc:.2f}%\n")

        # ⬇️ SAVE CHECKPOINT AFTER EACH EPOCH
        save_checkpoint(
            model, optimizer, epoch, current_step, best_val_acc,
            scheduler_warmup, scheduler_cosine, warmup_steps
        )

        log_file.flush()

    # ⬇️ SAVE FINAL MODEL TO SAGEMAKER OUTPUT PATH
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'model.pt'))
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'best_acc': best_val_acc,
    }, os.path.join(MODEL_DIR, 'model_full.pt'))

    print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📦 Model saved to {MODEL_DIR}")
    log_file.write(f"\nBest validation accuracy: {best_val_acc:.2f}%\n")
    log_file.close()

if __name__ == "__main__":

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
        parser.add_argument('--echo_batches', type=int, default=10, help='Print frequency')
        parser.add_argument('--val_interval', type=int, default=1, help='Validation interval')
        parser.add_argument('--num_workers', type=int, default=0, help='Dataloader workers')

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
        parser.add_argument('--v_dim', type=int, default=64)
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
        print(f"📂 Current Directory contents: {os.listdir('.')}", flush=True)
        if os.path.exists('models_comp'):
            print(f"📂 models_comp contents: {os.listdir('models_comp')}", flush=True)
        else:
            print("❌ ERROR: 'models_comp' folder NOT FOUND in container!", flush=True)

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