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



cv2.setNumThreads(0)
# ==========================================
# 1. INSTALL DEPENDENCIES
# ==========================================
def install_dependencies():
    # Check if already installed to avoid redundant calls in sub-processes
    try:
        import mediapipe as mp
        from tqdm import tqdm

    except ImportError:
        print("⚙️ Installing dependencies...")
        subprocess.check_call(["apt-get", "update"], stdout=subprocess.DEVNULL)
        subprocess.check_call(["apt-get", "install", "-y", "ffmpeg", "libgl1-mesa-glx", "wget"], stdout=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe", "pandas", "tqdm", "torchaudio"], stdout=subprocess.DEVNULL)


import mediapipe as mp
import torchaudio
from tqdm import tqdm


# ==========================================
# 2. PROCESSING FUNCTIONS
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
    # max_workers = max(1, multiprocessing.cpu_count() - 2)
    max_workers = 12
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
        
        results = []
        # EXECUTE PARALLEL PROCESSING
        # chunksize=4 helps reduce inter-process communication overhead
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # map the function to the list of arguments
            # results = list(executor.map(process_single_video, work_items, chunksize=4))
            process_iterator = executor.map(process_single_video, work_items, chunksize=4)
            
            for res in tqdm(process_iterator, total=len(work_items), desc=f"   Processing {split_name}", unit="vid"):
                results.append(res)
        

        # Summary
        success_count = results.count("SUCCESS")
        skip_count = results.count("SKIP")
        fail_count = len(results) - success_count - skip_count
        print(f"   ✅ Done {split_name}: {success_count} processed, {skip_count} skipped, {fail_count} failed.")

    print("🎉 All tasks finished.")
