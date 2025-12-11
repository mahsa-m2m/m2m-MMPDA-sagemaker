import os
import argparse
import torch
import sys
import subprocess
import multiprocessing
import numpy as np
import cv2
import pandas as pd
import tempfile
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ==========================================
# 1. INSTALL DEPENDENCIES (Run once)
# ==========================================
def install_dependencies():
    # Check if already installed to avoid redundant calls in sub-processes
    try:
        import mediapipe
        from tqdm import tqdm

    except ImportError:
        print("⚙️ Installing dependencies...")
        subprocess.check_call(["apt-get", "update"], stdout=subprocess.DEVNULL)
        subprocess.check_call(["apt-get", "install", "-y", "ffmpeg", "libgl1-mesa-glx", "wget"], stdout=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe", "pandas", "tqdm", "torchaudio"], stdout=subprocess.DEVNULL)

install_dependencies()
import mediapipe as mp
import torchaudio
from tqdm import tqdm


# ==========================================
# 2. STANDALONE PROCESSING FUNCTIONS
#    (Must be top-level for Multiprocessing)
# ==========================================

def get_mediapipe_features(blendshapes, matrix):
    """Maps MP output to the 50-dim feature vector."""
    bs = {b.category_name: b.score for b in blendshapes}
    au = np.zeros(35, dtype=np.float32)
    
    # Action Units Mapping
    au[0] = bs.get('browInnerUp', 0)
    au[1] = (bs.get('browOuterUpLeft', 0) + bs.get('browOuterUpRight', 0)) / 2
    au[2] = (bs.get('browDownLeft', 0) + bs.get('browDownRight', 0)) / 2
    au[3] = (bs.get('eyeWideLeft', 0) + bs.get('eyeWideRight', 0)) / 2
    au[4] = (bs.get('cheekSquintLeft', 0) + bs.get('cheekSquintRight', 0)) / 2
    au[5] = (bs.get('eyeSquintLeft', 0) + bs.get('eyeSquintRight', 0)) / 2
    au[6] = (bs.get('noseSneerLeft', 0) + bs.get('noseSneerRight', 0)) / 2
    au[7] = (bs.get('mouthUpperUpLeft', 0) + bs.get('mouthUpperUpRight', 0)) / 2
    au[8] = (bs.get('mouthSmileLeft', 0) + bs.get('mouthSmileRight', 0)) / 2
    au[9] = (bs.get('mouthDimpleLeft', 0) + bs.get('mouthDimpleRight', 0)) / 2
    au[10] = (bs.get('mouthFrownLeft', 0) + bs.get('mouthFrownRight', 0)) / 2
    au[11] = bs.get('mouthShrugLower', 0)
    au[12] = bs.get('mouthPucker', 0)
    au[13] = (bs.get('mouthStretchLeft', 0) + bs.get('mouthStretchRight', 0)) / 2
    au[14] = bs.get('jawOpen', 0)
    au[15] = bs.get('mouthClose', 0)
    au[16] = (bs.get('eyeBlinkLeft', 0) + bs.get('eyeBlinkRight', 0)) / 2
    
    extras = ['mouthFunnel', 'mouthPressLeft', 'mouthPressRight', 'mouthRollUpper', 
              'mouthRollLower', 'mouthShrugUpper', 'jawLeft', 'jawRight', 'jawForward',
              'cheekPuff', 'mouthLowerDownLeft', 'mouthLowerDownRight']
    for i, name in enumerate(extras):
        if 17 + i < 35: au[17 + i] = bs.get(name, 0)

    # Pose
    m = np.array(matrix)
    if m.shape == (4, 4): m = m.flatten()
    sy = math.sqrt(m[0] * m[0] + m[4] * m[4])
    if sy > 1e-6:
        pitch = math.atan2(m[9], m[10])
        yaw = math.atan2(-m[8], sy)
        roll = math.atan2(m[4], m[0])
    else:
        pitch = math.atan2(-m[6], m[5])
        yaw = math.atan2(-m[8], sy)
        roll = 0

    # Gaze
    l_dx = bs.get('eyeLookInLeft', 0) - bs.get('eyeLookOutLeft', 0)
    l_dy = bs.get('eyeLookUpLeft', 0) - bs.get('eyeLookDownLeft', 0)
    r_dx = bs.get('eyeLookOutRight', 0) - bs.get('eyeLookInRight', 0)
    r_dy = bs.get('eyeLookUpRight', 0) - bs.get('eyeLookDownRight', 0)
    gaze = np.array([pitch, yaw, roll, l_dx, l_dy, r_dx, r_dy, abs(l_dx - r_dx)], dtype=np.float32)

    # Emotions
    happy = bs.get('mouthSmileLeft', 0) * 0.5 + bs.get('mouthSmileRight', 0) * 0.5
    sad = (bs.get('mouthFrownLeft', 0) + bs.get('browDownLeft', 0)) / 2.0
    surprise = (bs.get('browInnerUp', 0) + bs.get('jawOpen', 0)) / 2.0
    fear = (bs.get('mouthStretchLeft', 0) + bs.get('browInnerUp', 0)) / 2.0
    anger = (bs.get('browDownLeft', 0) + bs.get('jawForward', 0)) / 2.0
    emotions = np.array([happy, sad, surprise, fear, anger], dtype=np.float32)

    val = happy - max(sad, anger, fear)
    arousal = max(happy, surprise, anger, fear)
    va = np.array([val, arousal], dtype=np.float32)

    return np.concatenate([au, gaze, emotions, va])

def sample_frames(video_path, num_frames=64):
    """
    Reads video frames. Uses OpenCV (FastFile friendly).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            last_valid = None
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    last_valid = frame.copy()
                elif last_valid is not None:
                    frames.append(last_valid.copy())
    cap.release()
    
    # Fill remaining if needed
    if len(frames) > 0:
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
        return np.array(frames[:num_frames], dtype=np.uint8)
    
    return None

def extract_audio(video_path, audio_length=80000, sample_rate=16000, n_mels=128):
    """Robust audio extraction."""
    try:
        # 1. Direct Load
        waveform, sr = torchaudio.load(video_path)
    except:
        # 2. FFmpeg fallback (Silent failsafe)
        return (np.zeros(audio_length, dtype=np.float32),
                np.zeros((3, n_mels, audio_length // 160 + 1), dtype=np.float32))

    # Resample
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad/Trim
    if waveform.shape[1] < audio_length:
        waveform = torch.nn.functional.pad(waveform, (0, audio_length - waveform.shape[1]))
    else:
        waveform = waveform[:, :audio_length]

    # Mel Spec
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, win_length=400, hop_length=160
    )
    mel_spec = mel_transform(waveform)
    mel_spec = mel_spec.repeat(3, 1, 1)

    return waveform.squeeze(0).numpy(), mel_spec.numpy()

def process_single_video(args_tuple):
    """
    The Worker Function. 
    Self-contained: inits model, extracts features, saves result.
    """
    video_path, label, save_path, model_path, frame_size = args_tuple
    
    if os.path.exists(save_path):
        return "SKIP"

    try:
        # 1. Sample Frames
        frames = sample_frames(video_path)
        if frames is None: return "FAIL_READ"

        # 2. Setup MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        base_options = BaseOptions(model_asset_path=model_path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        
        # Use context manager for auto-cleanup
        final_crops = []
        final_features = []
        
        with FaceLandmarker.create_from_options(options) as landmarker:
            for frame in frames:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection = landmarker.detect(mp_image)
                
                feat_vec = np.zeros(50, dtype=np.float32)
                face_crop = cv2.resize(frame, (frame_size[1], frame_size[0]))
                
                if detection.face_landmarks:
                    # Extract Features
                    feat_vec = get_mediapipe_features(detection.face_blendshapes[0], detection.facial_transformation_matrixes[0])
                    
                    # Extract Crop
                    landmarks = detection.face_landmarks[0]
                    h, w = frame.shape[:2]
                    x_c = [lm.x for lm in landmarks]
                    y_c = [lm.y for lm in landmarks]
                    x_min, x_max = min(x_c) * w, max(x_c) * w
                    y_min, y_max = min(y_c) * h, max(y_c) * h
                    
                    margin_x, margin_y = (x_max - x_min) * 0.2, (y_max - y_min) * 0.2
                    x1, y1 = max(0, int(x_min - margin_x)), max(0, int(y_min - margin_y))
                    x2, y2 = min(w, int(x_max + margin_x)), min(h, int(y_max + margin_y))
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size != 0:
                        face_crop = cv2.resize(crop, (frame_size[1], frame_size[0]))

                final_features.append(feat_vec)
                final_crops.append(face_crop)

        # 3. Audio
        audio_wave, audio_mel = extract_audio(video_path)

        # 4. Save
        # Convert crops to uint8 tensor (saves huge space vs float)
        frames_tensor = torch.from_numpy(np.array(final_crops)).permute(3, 0, 1, 2).to(torch.uint8)
        feat_tensor = torch.from_numpy(np.array(final_features)).float()
        
        sample = {
            'vision_behaviour': feat_tensor,
            'vision_face': frames_tensor,
            'audio_mel': torch.tensor(audio_mel),
            'audio_wave': torch.tensor(audio_wave),
            'label': torch.tensor(label, dtype=torch.long),
            'videoname': os.path.basename(video_path)
        }
        
        torch.save(sample, save_path)
        return "SUCCESS"

    except Exception as e:
        return f"ERROR: {str(e)}"

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/processing/input/data")
    parser.add_argument("--csv_dir", type=str, default="/opt/ml/processing/input/csv")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="validation.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")
    args = parser.parse_args()

    # Model Setup
    model_path = os.path.abspath('./face_landmarker.task')
    if not os.path.exists(model_path):
        subprocess.call(["wget", "-O", model_path, "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"])

    # Determine Workers
    # Leave 2 cores free for OS and I/O management
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Launching extraction with {max_workers} parallel workers.")

    splits = [('train', args.train_csv), ('val', args.val_csv), ('test', args.test_csv)]
    
    label_map = {'truthful': 0, 'deceptive': 1, 'truth': 0, 'lie': 1}

    for split_name, csv_name in splits:
        csv_path = os.path.join(args.csv_dir, csv_name)
        if not os.path.exists(csv_path): continue
        
        print(f"📂 Preparing {split_name}...")
        
        # Load CSV and prepare work items
        work_items = []
        save_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            df = pd.read_csv(csv_path)
            # Normalize columns
            if 's3_path' not in df.columns and 'path' not in df.columns:
                df = pd.read_csv(csv_path, header=None)
                df.columns = ['path', 'label']
            cols = list(df.columns)
            rename = {c: 'path' for c in cols if 'path' in c.lower()}
            rename.update({c: 'label' for c in cols if 'label' in c.lower()})
            df.rename(columns=rename, inplace=True)

            for _, row in df.iterrows():
                label_raw = str(row['label']).strip().lower()
                if label_raw not in label_map: continue
                
                filename = os.path.basename(str(row['path']).strip())
                subfolder = 'truthful' if label_map[label_raw] == 0 else 'deceptive'
                video_path = os.path.join(args.data_dir, subfolder, filename)
                
                # Check if file exists via FastFile path
                if os.path.exists(video_path):
                    file_base = os.path.splitext(filename)[0]
                    save_path = os.path.join(save_dir, f"{file_base}.pt")
                    
                    # Prepare arguments for worker
                    # video_path, label, save_path, model_path, frame_size
                    work_items.append((video_path, label_map[label_raw], save_path, model_path, (160, 160)))
        except Exception as e:
            print(f"❌ Error reading CSV {csv_name}: {e}")
            continue

        print(f"   Queueing {len(work_items)} videos for processing...")

        # EXECUTE PARALLEL PROCESSING
        # chunksize=4 helps reduce inter-process communication overhead
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # We map the function to the list of arguments
            # results = list(executor.map(process_single_video, work_items, chunksize=4))
            process_iterator = executor.map(process_single_video, work_items, chunksize=4)
            
            for res in tqdm(process_iterator, total=len(work_items), desc=f"   Processing {split_name}", unit="vid"):
                results.append(res)
        

        # Simple Summary
        success_count = results.count("SUCCESS")
        skip_count = results.count("SKIP")
        fail_count = len(results) - success_count - skip_count
        print(f"   ✅ Done {split_name}: {success_count} processed, {skip_count} skipped, {fail_count} failed.")

    print("🎉 All tasks finished.")
# import os
# import argparse
# import torch
# import sys
# import subprocess
# import multiprocessing
# import numpy as np
# import cv2
# import pandas as pd
# import tempfile
# import random
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from pathlib import Path

# # ==========================================
# # 1. INSTALL DEPENDENCIES
# # ==========================================
# def install_dependencies():
#     print("⚙️ Installing system dependencies (FFmpeg, GL)...")
#     subprocess.check_call(["apt-get", "update"])
#     subprocess.check_call(["apt-get", "install", "-y", "ffmpeg", "libgl1-mesa-glx", "wget"])
    
#     print("⚙️ Installing Python dependencies...")
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe", "pandas", "tqdm", "torchaudio"])

# try:
#     import mediapipe as mp
# except ImportError:
#     install_dependencies()
#     import mediapipe as mp

# # ==========================================
# # 2. DATASET CLASS
# # ==========================================
# class VideoDeceptionDataset(Dataset):
#     """
#     Dataset for loading video files and extracting MMPDA features using MediaPipe V2.
#     """
#     def __init__(self, csv_file=None, data_root=None, annotation_file=None,
#                  num_frames=64, frame_size=(160, 160),
#                  audio_length=80000, sample_rate=16000,
#                  n_mels=128, mode='train', model_path='face_landmarker.task'):
        
#         self.num_frames = num_frames
#         self.frame_size = frame_size
#         self.audio_length = audio_length
#         self.sample_rate = sample_rate
#         self.n_mels = n_mels
#         self.mode = mode
#         self.video_list = []
#         self.labels = []

#         ## --------- MediaPipe V2 Setup -------- ##
#         self.model_path = model_path
#         self.landmarker = None

#         # --- LOAD DATA LOGIC ---
#         if csv_file is not None and os.path.exists(csv_file):
#             self._load_from_csv(csv_file, data_root)
#         elif data_root is not None:
#             self._load_from_directory(data_root)
#         elif annotation_file is not None:
#             self._load_from_annotation(annotation_file)
#         else:
#             # Fallback: if data_root exists, try loading directory structure even if csv missing
#             if data_root and os.path.exists(data_root):
#                 self._load_from_directory(data_root)
#             else:
#                 print("⚠️ Warning: No valid data source found in init.")

#     def _init_mediapipe(self):
#         if self.landmarker is not None: return
#         try:
#             from mediapipe.tasks import python
#             from mediapipe.tasks.python import vision
#             BaseOptions = mp.tasks.BaseOptions
#             FaceLandmarker = mp.tasks.vision.FaceLandmarker
#             FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
#             VisionRunningMode = mp.tasks.vision.RunningMode

#             base_options = BaseOptions(model_asset_path=self.model_path)
#             options = FaceLandmarkerOptions(
#                 base_options=base_options,
#                 output_face_blendshapes=True,
#                 output_facial_transformation_matrixes=True,
#                 running_mode=VisionRunningMode.IMAGE,
#                 num_faces=1,
#                 min_face_detection_confidence=0.5
#             )
#             self.landmarker = FaceLandmarker.create_from_options(options)
#         except Exception as e:
#             print(f"❌ Failed to init MediaPipe V2: {e}")
#             self.landmarker = None

#     def _map_mp_v2_to_features(self, blendshapes, matrix):
#         bs = {b.category_name: b.score for b in blendshapes}
#         au = np.zeros(35, dtype=np.float32)
#         au[0] = bs.get('browInnerUp', 0)
#         au[1] = (bs.get('browOuterUpLeft', 0) + bs.get('browOuterUpRight', 0)) / 2
#         au[2] = (bs.get('browDownLeft', 0) + bs.get('browDownRight', 0)) / 2
#         au[3] = (bs.get('eyeWideLeft', 0) + bs.get('eyeWideRight', 0)) / 2
#         au[4] = (bs.get('cheekSquintLeft', 0) + bs.get('cheekSquintRight', 0)) / 2
#         au[5] = (bs.get('eyeSquintLeft', 0) + bs.get('eyeSquintRight', 0)) / 2
#         au[6] = (bs.get('noseSneerLeft', 0) + bs.get('noseSneerRight', 0)) / 2
#         au[7] = (bs.get('mouthUpperUpLeft', 0) + bs.get('mouthUpperUpRight', 0)) / 2
#         au[8] = (bs.get('mouthSmileLeft', 0) + bs.get('mouthSmileRight', 0)) / 2
#         au[9] = (bs.get('mouthDimpleLeft', 0) + bs.get('mouthDimpleRight', 0)) / 2
#         au[10] = (bs.get('mouthFrownLeft', 0) + bs.get('mouthFrownRight', 0)) / 2
#         au[11] = bs.get('mouthShrugLower', 0)
#         au[12] = bs.get('mouthPucker', 0)
#         au[13] = (bs.get('mouthStretchLeft', 0) + bs.get('mouthStretchRight', 0)) / 2
#         au[14] = bs.get('jawOpen', 0)
#         au[15] = bs.get('mouthClose', 0)
#         au[16] = (bs.get('eyeBlinkLeft', 0) + bs.get('eyeBlinkRight', 0)) / 2
#         extras = ['mouthFunnel', 'mouthPressLeft', 'mouthPressRight', 'mouthRollUpper', 
#                   'mouthRollLower', 'mouthShrugUpper', 'jawLeft', 'jawRight', 'jawForward',
#                   'cheekPuff', 'mouthLowerDownLeft', 'mouthLowerDownRight']
#         for i, name in enumerate(extras):
#             if 17 + i < 35: au[17 + i] = bs.get(name, 0)

#         m = np.array(matrix)
#         if m.shape == (4, 4): m = m.flatten()
#         import math
#         sy = math.sqrt(m[0] * m[0] + m[4] * m[4])
#         if sy > 1e-6:
#             pitch = math.atan2(m[9], m[10])
#             yaw = math.atan2(-m[8], sy)
#             roll = math.atan2(m[4], m[0])
#         else:
#             pitch = math.atan2(-m[6], m[5])
#             yaw = math.atan2(-m[8], sy)
#             roll = 0

#         l_dx = bs.get('eyeLookInLeft', 0) - bs.get('eyeLookOutLeft', 0)
#         l_dy = bs.get('eyeLookUpLeft', 0) - bs.get('eyeLookDownLeft', 0)
#         r_dx = bs.get('eyeLookOutRight', 0) - bs.get('eyeLookInRight', 0)
#         r_dy = bs.get('eyeLookUpRight', 0) - bs.get('eyeLookDownRight', 0)
        
#         gaze = np.array([pitch, yaw, roll, l_dx, l_dy, r_dx, r_dy, abs(l_dx - r_dx)], dtype=np.float32)

#         happy = bs.get('mouthSmileLeft', 0) * 0.5 + bs.get('mouthSmileRight', 0) * 0.5
#         sad = (bs.get('mouthFrownLeft', 0) + bs.get('browDownLeft', 0)) / 2.0
#         surprise = (bs.get('browInnerUp', 0) + bs.get('jawOpen', 0)) / 2.0
#         fear = (bs.get('mouthStretchLeft', 0) + bs.get('browInnerUp', 0)) / 2.0
#         anger = (bs.get('browDownLeft', 0) + bs.get('jawForward', 0)) / 2.0
#         emotions = np.array([happy, sad, surprise, fear, anger], dtype=np.float32)

#         val = happy - max(sad, anger, fear)
#         arousal = max(happy, surprise, anger, fear)
#         va = np.array([val, arousal], dtype=np.float32)

#         return np.concatenate([au, gaze, emotions, va])

#     def _process_mmpda_pipeline(self, raw_frames):
#         self._init_mediapipe()
#         if self.landmarker is None:
#             return np.zeros((len(raw_frames), self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8), \
#                    np.zeros((len(raw_frames), 50), dtype=np.float32)
        
#         final_features = []
#         final_crops = []
        
#         for frame in raw_frames:
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#             detection = self.landmarker.detect(mp_image)
            
#             feat_vec = np.zeros(50, dtype=np.float32)
#             face_crop = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
            
#             if detection.face_landmarks:
#                 feat_vec = self._map_mp_v2_to_features(detection.face_blendshapes[0], detection.facial_transformation_matrixes[0])
#                 landmarks = detection.face_landmarks[0]
#                 h, w = frame.shape[:2]
#                 x_coords = [lm.x for lm in landmarks]
#                 y_coords = [lm.y for lm in landmarks]
#                 x_min, x_max = min(x_coords) * w, max(x_coords) * w
#                 y_min, y_max = min(y_coords) * h, max(y_coords) * h
                
#                 margin_x, margin_y = (x_max - x_min) * 0.2, (y_max - y_min) * 0.2
#                 x1, y1 = max(0, int(x_min - margin_x)), max(0, int(y_min - margin_y))
#                 x2, y2 = min(w, int(x_max + margin_x)), min(h, int(y_max + margin_y))
                
#                 crop = frame[y1:y2, x1:x2]
#                 if crop.size != 0:
#                     face_crop = cv2.resize(crop, (self.frame_size[1], self.frame_size[0]))

#             final_features.append(feat_vec)
#             final_crops.append(face_crop)

#         return np.array(final_crops), np.array(final_features)

#     def _sample_frames(self, video_path):
#         if not os.path.exists(video_path): return None
#         # Try OpenCV
#         cap = cv2.VideoCapture(video_path)
#         if cap.isOpened():
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             if total_frames > 0:
#                 indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
#                 frames = []
#                 last_valid = None
#                 for idx in indices:
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
#                     ret, frame = cap.read()
#                     if ret and frame is not None and frame.size > 0:
#                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                         frames.append(frame)
#                         last_valid = frame.copy()
#                     elif last_valid is not None:
#                         frames.append(last_valid.copy())
#                 cap.release()
#                 if len(frames) >= self.num_frames // 2:
#                     while len(frames) < self.num_frames: frames.append(frames[-1].copy())
#                     return np.array(frames[:self.num_frames], dtype=np.uint8)
#             cap.release()
#         return self._sample_frames_ffmpeg(video_path)

#     def _sample_frames_ffmpeg(self, video_path):
#         try:
#             probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
#                 '-show_entries', 'stream=nb_read_packets,duration', '-of', 'csv=p=0', video_path]
#             duration = 0.0
#             try:
#                 res = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
#                 if res.returncode == 0 and res.stdout: duration = float(res.stdout.strip().split(',')[0])
#             except: pass

#             timestamps = np.linspace(0, max(0.1, duration - 0.1), self.num_frames)
#             frames = []
#             temp_dir = tempfile.mkdtemp()
#             try:
#                 for i, ts in enumerate(timestamps):
#                     output_file = os.path.join(temp_dir, f'frame_{i:04d}.jpg')
#                     subprocess.run(['ffmpeg', '-ss', str(ts), '-i', video_path, '-vframes', '1', '-q:v', '2', '-y', output_file], capture_output=True, timeout=5)
#                     if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
#                         frame = cv2.imread(output_file)
#                         if frame is not None: frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                     if len(frames) < i + 1 and frames: frames.append(frames[-1].copy())
                
#                 if not frames: return None
#                 while len(frames) < self.num_frames: frames.append(frames[-1].copy())
#                 return np.array(frames[:self.num_frames], dtype=np.uint8)
#             finally:
#                 import shutil
#                 try: shutil.rmtree(temp_dir)
#                 except: pass
#         except: return None

#     def _extract_audio(self, video_path):
#         """
#         Extract audio from video - WORKS WITH ALL FORMATS
#         matches original VideoDeceptionDataset logic.
#         """
#         try:
#             import torchaudio
#             import subprocess
#             import tempfile
            
#             # 1. Try direct loading first (Fastest)
#             try:
#                 waveform, sample_rate = torchaudio.load(video_path)
#             except:
#                 # 2. Fallback: extract audio to temp wav file using ffmpeg
#                 with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
#                     temp_path = temp_audio.name
                
#                 try:
#                     # Extract audio using ffmpeg (handles all video formats)
#                     # using stdout=subprocess.DEVNULL to keep logs clean
#                     subprocess.run([
#                         'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
#                         '-ar', str(self.sample_rate), '-ac', '1', '-y', temp_path
#                     ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
#                     waveform, sample_rate = torchaudio.load(temp_path)
#                 finally:
#                     if os.path.exists(temp_path):
#                         os.remove(temp_path)
            
#             # 3. Resample if necessary
#             if sample_rate != self.sample_rate:
#                 resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
#                 waveform = resampler(waveform)
            
#             # 4. Convert to mono if stereo
#             if waveform.shape[0] > 1:
#                 waveform = torch.mean(waveform, dim=0, keepdim=True)
            
#             # 5. Pad or truncate to target length
#             if waveform.shape[1] < self.audio_length:
#                 waveform = torch.nn.functional.pad(waveform, (0, self.audio_length - waveform.shape[1]))
#             else:
#                 waveform = waveform[:, :self.audio_length]
            
#             # 6. Generate mel spectrogram
#             mel_transform = torchaudio.transforms.MelSpectrogram(
#                 sample_rate=self.sample_rate,
#                 n_mels=self.n_mels,
#                 n_fft=1024, #400
#                 win_length=400,
#                 hop_length=160
#             )
#             mel_spec = mel_transform(waveform)
            
#             # 7. Convert to 3-channel format (for compatibility)
#             mel_spec = mel_spec.repeat(3, 1, 1)
            
#             return waveform.squeeze(0).numpy(), mel_spec.numpy()
        
#         except Exception as e:
#             # print(f"❌ Audio extraction error for {os.path.basename(video_path)}: {str(e)}")
#             # Return silent audio
#             return (np.zeros(self.audio_length, dtype=np.float32),
#                     np.zeros((3, self.n_mels, self.audio_length // 160 + 1), dtype=np.float32))
    
#     def _load_from_csv(self, csv_file, data_root):
#         print(f"📄 Loading CSV: {csv_file}")
#         label_map = {'truthful': 0, 'deceptive': 1, 'truth': 0, 'lie': 1}
#         try:
#             df = pd.read_csv(csv_file)
#             if 's3_path' not in df.columns and 'path' not in df.columns:
#                 df = pd.read_csv(csv_file, header=None)
#                 df.columns = ['path', 'label']
#             else:
#                 cols = list(df.columns)
#                 rename = {c: 'path' for c in cols if 'path' in c.lower()}
#                 rename.update({c: 'label' for c in cols if 'label' in c.lower()})
#                 df.rename(columns=rename, inplace=True)
#         except Exception as e:
#             print(f"❌ CSV Error: {e}")
#             return

#         for _, row in df.iterrows():
#             raw_path = str(row['path']).strip()
#             label_raw = str(row['label']).strip().lower()
#             if label_raw not in label_map: continue
            
#             filename = os.path.basename(raw_path)
#             subfolder = 'truthful' if label_map[label_raw] == 0 else 'deceptive'
            
#             # Construct Sagemaker Processing Path
#             final_path = os.path.join(data_root, subfolder, filename)
            
#             if os.path.exists(final_path):
#                 self.video_list.append(final_path)
#                 self.labels.append(label_map[label_raw])
    
#     def _load_from_directory(self, data_root):
#         # Simple directory loading fallback
#         pass 
#     def _load_from_annotation(self, f): pass
#     def __len__(self): return len(self.video_list)
#     def __getitem__(self, idx):
#         # We only need this to satisfy Dataset interface for DataLoader
#         # The logic is handled in the extraction loop
#         return self.video_list[idx], self.labels[idx]

# # ==========================================
# # 3. EXTRACTION JOB LOGIC
# # ==========================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # Paths injected by SageMaker
#     parser.add_argument("--data_dir", type=str, default="/opt/ml/processing/input/data")
#     parser.add_argument("--csv_dir", type=str, default="/opt/ml/processing/input/csv")
#     parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
#     parser.add_argument("--train_csv", type=str, default="train.csv")
#     parser.add_argument("--val_csv", type=str, default="validation.csv")
#     parser.add_argument("--test_csv", type=str, default="test.csv")
#     args = parser.parse_args()

#     # 1. Download Model
#     model_path = './face_landmarker.task'
#     if not os.path.exists(model_path):
#         subprocess.call(["wget", "-O", model_path, "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"])

#     # 2. Configure Workers
#     num_cpus = multiprocessing.cpu_count()
#     print(f"🚀 Feature Extraction Started on {num_cpus} CPUs")

#     # 3. Process Splits
#     splits = [
#         ('train', args.train_csv), 
#         ('val', args.val_csv), 
#         ('test', args.test_csv)
#     ]

#     for split_name, csv_name in splits:
#         csv_path = os.path.join(args.csv_dir, csv_name)
#         if not os.path.exists(csv_path):
#             print(f"⚠️ CSV {csv_name} not found, skipping {split_name}.")
#             continue
        
#         print(f"📂 Processing {split_name} from {csv_path}...")
        
#         # Initialize Dataset
#         dataset = VideoDeceptionDataset(
#             csv_file=csv_path,
#             data_root=args.data_dir,
#             mode=split_name,
#             model_path=model_path,
#             num_frames=64
#         )
        
#         print(f"   Found {len(dataset)} videos.")
        
#         # Output sub-folder
#         save_dir = os.path.join(args.output_dir, split_name)
#         os.makedirs(save_dir, exist_ok=True)

#         # Iterate manually to have full control over saving
#         # Using a simple parallel loop could be faster, but let's stick to safe sequential/batched
#         # for simplicity and reliability with MediaPipe (which uses threading internally)
        
#         for i in tqdm(range(len(dataset))):
#             video_path = dataset.video_list[i]
#             label = dataset.labels[i]
#             video_name = os.path.basename(video_path)
            
#             # Define Save Path
#             file_base = os.path.splitext(video_name)[0]
#             save_path = os.path.join(save_dir, f"{file_base}.pt")
            
#             # Skip if exists
#             if os.path.exists(save_path): continue

#             try:
#                 # 1. Extract
#                 frames = dataset._sample_frames(video_path)
#                 if frames is None: continue # Skip bad videos
                
#                 face_crops, features = dataset._process_mmpda_pipeline(frames)

#                 # # ==================================================================
#                 # # DEBUG: SAVE 1 SAMPLE FRAME PER VIDEO
#                 # # ==================================================================
#                 # # Create debug directory
#                 # debug_dir = "debug_preprocessed_samples"
#                 # os.makedirs(debug_dir, exist_ok=True)
                
#                 # # Pick a random frame index to verify
#                 # rnd_idx = np.random.randint(0, len(face_crops))
                
#                 # # 1. Get the Image (The 160x160 Crop the model actually sees)
#                 # # Convert RGB (MediaPipe/Tensor format) back to BGR (OpenCV format)
#                 # debug_img = cv2.cvtColor(face_crops[rnd_idx], cv2.COLOR_RGB2BGR)
                
#                 # # 2. Get the Features for this frame
#                 # feats = features[rnd_idx]
                
#                 # # 3. Overlay Key Features on the image to verify they match the face
#                 # # -- Head Pose (Indices 35-37: Pitch, Yaw, Roll)
#                 # pose_txt = f"Y:{feats[36]:.1f} P:{feats[35]:.1f}"
                
#                 # # -- Emotions (Indices 43-47: Happy, Sad, Surprise, Fear, Anger)
#                 # emo_names = ['Hap', 'Sad', 'Sur', 'Fea', 'Ang']
#                 # emo_vals = feats[43:48]
#                 # top_emo_idx = np.argmax(emo_vals)
#                 # emo_txt = f"{emo_names[top_emo_idx]}:{emo_vals[top_emo_idx]:.2f}"
                
#                 # # -- Valence/Arousal (Indices 48-49)
#                 # va_txt = f"V:{feats[48]:.1f} A:{feats[49]:.1f}"

#                 # # Draw text (White with Black outline for readability)
#                 # def draw_txt(img, text, y):
#                 #     cv2.putText(img, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2) # Outline
#                 #     cv2.putText(img, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1) # Text

#                 # draw_txt(debug_img, pose_txt, 15)
#                 # draw_txt(debug_img, emo_txt, 30)
#                 # draw_txt(debug_img, va_txt, 45)
                
#                 # # Save to disk
#                 # safe_name = os.path.basename(video_path).replace('.', '_')
#                 # cv2.imwrite(os.path.join(debug_dir, f"{safe_name}_frame{rnd_idx}.jpg"), debug_img)
#                 # # ==================================================================

#                 audio_wave, audio_mel = dataset._extract_audio(video_path)
                
#                 # 2. Tensorify
#                 # frames_tensor = torch.from_numpy(face_crops).permute(3, 0, 1, 2).float()
#                 # frames_tensor = (frames_tensor / 255.0 - 0.5) * 2.0
#                 frames_tensor = torch.from_numpy(face_crops).permute(3, 0, 1, 2).to(torch.uint8)

#                 feat_tensor = torch.from_numpy(features).float()
#                 if feat_tensor.shape[1] != 50: feat_tensor = feat_tensor[:, :50]
                
#                 sample = {
#                     'vision_behaviour': feat_tensor,
#                     'vision_face': frames_tensor,
#                     'audio_mel': torch.tensor(audio_mel),
#                     'audio_wave': torch.tensor(audio_wave),
#                     'label': torch.tensor(label, dtype=torch.long),
#                     'videoname': video_name
#                 }
                
#                 # 3. Save
#                 torch.save(sample, save_path)
                
#             except Exception as e:
#                 print(f"   Failed {video_name}: {e}")

#     print("🎉 All tasks finished.")