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


SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', './model')
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', './output')

# ==================== COMPREHENSIVE WARNING SUPPRESSION ====================

# 1. Suppress Python warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 2. Suppress TensorFlow/MediaPipe logs (must be set BEFORE importing mediapipe)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only
os.environ['GLOG_minloglevel'] = '3'  # Google logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 3. Suppress OpenCV/FFmpeg warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '0'
cv2.setLogLevel(0)  # 0 = silent


# 4. Redirect stderr temporarily for library initialization
class SuppressOutput:
    def __enter__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
        return self

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


# Import with suppression
with SuppressOutput():
    # 5. Suppress ABSL logging (MediaPipe uses this)
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)

    import mediapipe as mp

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

        # Load from directory structure
        if data_root is not None:
            self._load_from_directory(data_root)
        # Load from annotation file
        elif annotation_file is not None:
            self._load_from_annotation(annotation_file)
        else:
            raise ValueError("Either data_root or annotation_file must be provided")

    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh (called in each worker process)"""
        if self.face_mesh is None:
            with SuppressOutput():
                mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=3, # 1
                    refine_landmarks=True,
                    min_detection_confidence=0.4, # 0.5
                    min_tracking_confidence=0.4 # 0.5
                )
                # Warm up the model with a dummy image to complete initialization
                dummy_image = np.zeros((160, 160, 3), dtype=np.uint8)
                _ = self.face_mesh.process(dummy_image)

    def _load_from_directory(self, data_root):
        """Load videos from directory structure"""
        data_root = Path(data_root)

        # Load truthful videos (label 0)
        truthful_dir = data_root / 'truthful'
        if truthful_dir.exists():
            for video_file in truthful_dir.glob('*'):
                if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    self.video_list.append(str(video_file))
                    self.labels.append(0)

        # Load deceptive videos (label 1)
        deceptive_dir = data_root / 'deceptive'
        if deceptive_dir.exists():
            for video_file in deceptive_dir.glob('*'):
                if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    self.video_list.append(str(video_file))
                    self.labels.append(1)

        print(f"Loaded {len(self.video_list)} videos from {data_root}")
        print(f"Truthful: {self.labels.count(0)}, Deceptive: {self.labels.count(1)}")

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

    def _extract_audio(self, video_path):
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

    def _sample_frames(self, video_path):
        """
        Sample frames uniformly from video with proper error handling
        FIX: Use FFmpeg fallback for videos that OpenCV can't read (especially MKV files)
        """
        # First attempt: OpenCV direct reading
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            # Fallback: Use FFmpeg to pipe frames directly
            return self._sample_frames_ffmpeg(video_path)

        # Set backend to FFmpeg for better H.264 handling
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # FIX: If frame count is unreliable (common with H.264), estimate from duration
        if total_frames <= 0 or fps <= 0:
            cap.release()
            return self._sample_frames_ffmpeg(video_path)

        # Normal case: known frame count
        if total_frames < self.num_frames:
            frame_indices = np.linspace(0, max(0, total_frames - 1), self.num_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        failed_reads = 0
        max_failures = self.num_frames // 4  # Allow up to 25% read failures

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret and frame is not None:
                try:
                    # Resize frame
                    frame = cv2.resize(frame, self.frame_size)
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    failed_reads = 0  # Reset failure counter on success
                except Exception as e:
                    failed_reads += 1
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))
            else:
                failed_reads += 1
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))

            # If too many failures, fall back to FFmpeg
            if failed_reads > max_failures:
                cap.release()
                return self._sample_frames_ffmpeg(video_path)

        cap.release()

        # Ensure we have exactly num_frames
        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))

        # Stack frames: (T, H, W, C)
        frames = np.stack(frames[:self.num_frames], axis=0)
        return frames

    def _sample_frames_ffmpeg(self, video_path):
        """
        Fallback method: Use FFmpeg directly to extract frames
        This handles MKV and other problematic formats that OpenCV can't read
        """
        try:
            # First, get video duration and fps using ffprobe
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration,nb_frames,r_frame_rate',
                '-of', 'default=noprint_wrappers=1',
                video_path
            ]

            probe_result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse output
            duration = None
            nb_frames = None
            fps = None

            for line in probe_result.stdout.split('\n'):
                if 'duration=' in line:
                    try:
                        duration = float(line.split('=')[1])
                    except:
                        pass
                elif 'nb_frames=' in line:
                    try:
                        nb_frames = int(line.split('=')[1])
                    except:
                        pass
                elif 'r_frame_rate=' in line:
                    try:
                        rate_parts = line.split('=')[1].split('/')
                        fps = float(rate_parts[0]) / float(rate_parts[1])
                    except:
                        pass

            # Estimate total frames
            if nb_frames:
                total_frames = nb_frames
            elif duration and fps:
                total_frames = int(duration * fps)
            else:
                total_frames = self.num_frames  # Fallback

            # Calculate frame indices to extract
            if total_frames < self.num_frames:
                frame_indices = list(range(total_frames))
                # Duplicate last frame if needed
                while len(frame_indices) < self.num_frames:
                    frame_indices.append(frame_indices[-1] if frame_indices else 0)
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

            frames = []

            # Extract frames using FFmpeg
            for idx in frame_indices:
                # Seek to specific frame and extract one frame
                cmd = [
                    'ffmpeg',
                    '-ss', str(idx / max(fps, 1)),  # Seek to time position
                    '-i', video_path,
                    '-vframes', '1',  # Extract 1 frame
                    '-f', 'rawvideo',
                    '-pix_fmt', 'rgb24',
                    '-s', f'{self.frame_size[0]}x{self.frame_size[1]}',
                    '-v', 'quiet',
                    '-'
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=5
                )

                if result.returncode == 0 and len(result.stdout) > 0:
                    # Parse raw RGB data
                    expected_size = self.frame_size[0] * self.frame_size[1] * 3
                    if len(result.stdout) >= expected_size:
                        frame_data = np.frombuffer(result.stdout[:expected_size], dtype=np.uint8)
                        frame = frame_data.reshape((self.frame_size[1], self.frame_size[0], 3))
                        frames.append(frame)
                    else:
                        if frames:
                            frames.append(frames[-1].copy())
                        else:
                            frames.append(np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8))
                else:
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8))

            # Ensure correct number of frames
            while len(frames) < self.num_frames:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8))

            frames = np.stack(frames[:self.num_frames], axis=0)
            return frames

        except Exception as e:
            if self.mode == 'train' and random.random() < 0.1:
                print(f"FFmpeg fallback also failed for {os.path.basename(video_path)}: {str(e)[:50]}")
            # Return black frames as last resort
            return np.zeros((self.num_frames, self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)

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

        # # 🔍 VALIDATION
        # assert features.shape == (64,), f"Expected 64 features, got {features.shape}"
        # assert not np.any(np.isnan(features)), "NaN values in features!"
        # assert not np.any(np.isinf(features)), "Inf values in features!"

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

        # Extract face frames
        frames = self._sample_frames(video_path)
        if frames is None:
            frames = np.zeros((self.num_frames, self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)

        # Extract audio
        audio_wave, audio_mel = self._extract_audio(video_path)

        # Extract behavioral features (OpenFace + Affect)
        behavioral_features = self._extract_behavioral_features(frames)

        # Convert face frames to tensor: (T, H, W, C) -> (C, T, H, W) for the model
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        # Normalize to [-1, 1]
        frames = (frames / 255.0 - 0.5) * 2.0

        # 🔍 DEBUG: Print shapes
        if idx == 0:  # Print for first sample
            print(f"\n🔍 DEBUG Info for video: {os.path.basename(video_path)}")
            print(f"  Frames shape: {frames.shape}")
            print(f"  Behavioral features shape: {behavioral_features.shape}")
            print(f"  Behavioral features sample: {behavioral_features[0][:10]}")  # First 10 features
            print(f"  Non-zero features: {np.count_nonzero(behavioral_features)}/{behavioral_features.size}")
        
        # Convert behavioral features to tensor
        behavioral_features = torch.from_numpy(behavioral_features).float()

        # 🔍 DEBUG: Print tensor shapes
        if idx == 0:
            print(f"  Behavioral tensor shape: {behavioral_features.shape}")
            print(f"  Expected shape: ({self.num_frames}, 64)\n")
        
        sample = {
            'vision_behaviour': behavioral_features,  # (T=64, 64)
            'vision_face': frames,  # (C=3, T=64, H=160, W=160)
            'audio_mel': audio_mel,  # (C=3, n_mels=128, time)
            'audio_wave': audio_wave,  # (audio_length,)
            'label': torch.tensor(label, dtype=torch.long),
            'videoname': os.path.basename(video_path)
        }

        return sample


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


def main(args):
    # Setup
    setup_seed(42)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Create log directory
    os.makedirs(args.log, exist_ok=True)
    os.makedirs(SM_MODEL_DIR, exist_ok=True)  # ← Add this

    log_file = open(os.path.join(args.log, 'training_log.txt'), 'w')

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
        num_workers=0,  # Set to 0 if MediaPipe issues persist
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=0,  # Set to 0 if MediaPipe issues persist
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

    # Training loop
    best_val_acc = 0.0
    current_step = 0

    for epoch in range(args.max_epochs):
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
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_val_acc,
                    # 'args': args,
                }, os.path.join(args.log, 'best_model.pt'))
                
                # 2. Save to SM_MODEL_DIR (SageMaker uploads this!)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_val_acc,
                }, os.path.join(SM_MODEL_DIR, 'best_model.pt'))
                
                print(f"✓ Saved best model to {SM_MODEL_DIR} with accuracy: {best_val_acc:.2f}%")
                print(f"✓ Saved best model with accuracy: {best_val_acc:.2f}%")
                log_file.write(f"Saved best model with accuracy: {best_val_acc:.2f}%\n")

        log_file.flush()
    
    # IMPORTANT: Save final model at the end too
    print(f"\nSaving final model to {SM_MODEL_DIR}...")
    torch.save({
        'epoch': args.max_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_val_acc,
    }, os.path.join(SM_MODEL_DIR, 'model.pt'))

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    log_file.write(f"\nBest validation accuracy: {best_val_acc:.2f}%\n")
    log_file.close()


if __name__ == "__main__":
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

    # Validate arguments
    if args.train_root is None and args.train_list is None:
        parser.error("Either --train_root or --train_list must be provided")
    if args.val_root is None and args.val_list is None:
        parser.error("Either --val_root or --val_list must be provided")

    main(args)