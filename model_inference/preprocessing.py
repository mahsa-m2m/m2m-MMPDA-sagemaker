import os
import cv2
import torch
import torchaudio
import numpy as np
import mediapipe as mp
import subprocess
import tempfile
import random
import config
import io

class InferencePreprocessor:
    def __init__(self):
        # 1. Store the class reference
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None  # Crucial: Start as None
        
        # Audio Transforms (Safe to pickle/copy across workers)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_mels=config.N_MELS,
            n_fft=400,
            hop_length=160
        )

    # --- PICKLE PROTECTION  ---
    def __getstate__(self):
        """Called when PyTorch sends this object to a worker process."""
        state = self.__dict__.copy()
        state['face_mesh'] = None 
        return state

    def __setstate__(self, state):
        """Called when the worker process receives this object."""
        self.__dict__.update(state)
        # Ensure it starts as None in the new process
        self.face_mesh = None 

    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh (Called only when needed inside the worker)"""
        if self.face_mesh is None:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1, 
                refine_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            )

    def process_video(self, video_path):
        """
        Main entry point: Converts video -> Dictionary of Tensors
        """
        # 2. Initialize MediaPipe
        # For num_workers > 0
        self._init_mediapipe()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Sample Frames
        frames_np = self._sample_frames(video_path) 
        
        # Extract Behavioral Features
        behavioral_np = self._extract_behavioral_features(frames_np)

        # Extract Audio
        audio_wave_np, audio_mel_np = self._extract_audio(video_path)

        # Convert to Tensors
        vision_face = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()
        vision_face = (vision_face / 255.0 - 0.5) * 2.0
        
        vision_behaviour = torch.from_numpy(behavioral_np).float()
        
        audio_wave = torch.from_numpy(audio_wave_np).float()
        audio_mel = torch.from_numpy(audio_mel_np).float()

        # Add Batch Dimension (B=1)
        return {
            'vision_behaviour': vision_behaviour.unsqueeze(0),
            'vision_face': vision_face.unsqueeze(0),
            'audio_mel': audio_mel.unsqueeze(0),
            'audio_wave': audio_wave.unsqueeze(0)
        }

    def _extract_behavioral_features(self, frames):
        """Run MediaPipe on frames"""
        # Double check initialization
        self._init_mediapipe()
        
        all_feats = []
        for frame in frames:
            results = self.face_mesh.process(frame)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                feats = self._compute_features_from_landmarks(lm, frame.shape) 
            else:
                feats = np.zeros(50, dtype=np.float32)
            all_feats.append(feats)
            
        return np.stack(all_feats)

    # --- HELPER FUNCTIONS ---

    # def _extract_audio(self, video_path):
    #     """Extracts audio with silence fallback"""
    #     temp_wav = None
    #     try:
    #         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
    #             temp_wav = tf.name
            
    #         cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
    #                '-ar', str(config.SAMPLE_RATE), '-ac', '1', '-y', temp_wav]
    #         subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)

    #         if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
    #             waveform, sr = torchaudio.load(temp_wav)
    #         else:
    #             raise ValueError("Empty audio")

    #         if waveform.shape[1] < config.AUDIO_LENGTH:
    #             waveform = torch.nn.functional.pad(waveform, (0, config.AUDIO_LENGTH - waveform.shape[1]))
    #         else:
    #             waveform = waveform[:, :config.AUDIO_LENGTH]

    #         mel_spec = self.mel_transform(waveform)
    #         mel_spec = mel_spec.repeat(3, 1, 1)

    #         if os.path.exists(temp_wav): os.remove(temp_wav)
    #         return waveform.squeeze(0).numpy(), mel_spec.numpy()

    #     except Exception as e:
    #         if temp_wav and os.path.exists(temp_wav): os.remove(temp_wav)
    #         return self._get_silent_audio()
    def _extract_audio(self, video_path):
        """Extracts audio directly to memory (No Disk I/O)"""
        try:
            # ffmpeg command to pipe audio as WAV to stdout
            cmd = [
                'ffmpeg', 
                '-i', video_path, 
                '-vn',               # No video
                '-f', 'wav',         # Format wav
                '-acodec', 'pcm_s16le', 
                '-ar', str(config.SAMPLE_RATE), 
                '-ac', '1',          # Mono
                '-loglevel', 'error', # Quieter output
                'pipe:1'             # Output to stdout
            ]
            
            # Run command and capture output in memory
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Read from memory buffer (No disk latency)
            memory_file = io.BytesIO(process.stdout)
            waveform, sr = torchaudio.load(memory_file)

            if waveform.shape[1] == 0:
                raise ValueError("Empty audio")

            # Exact same padding logic as before
            if waveform.shape[1] < config.AUDIO_LENGTH:
                waveform = torch.nn.functional.pad(waveform, (0, config.AUDIO_LENGTH - waveform.shape[1]))
            else:
                waveform = waveform[:, :config.AUDIO_LENGTH]

            mel_spec = self.mel_transform(waveform)
            mel_spec = mel_spec.repeat(3, 1, 1)

            return waveform.squeeze(0).numpy(), mel_spec.numpy()

        except Exception as e:
            # Fallback to silence if audio fails
            return self._get_silent_audio()

    def _get_silent_audio(self):
        silent_wave = np.zeros(config.AUDIO_LENGTH, dtype=np.float32)
        n_time_steps = (config.AUDIO_LENGTH // 160) + 1
        silent_mel = np.zeros((3, config.N_MELS, n_time_steps), dtype=np.float32)
        return silent_wave, silent_mel

    # def _sample_frames(self, video_path):
    #     cap = cv2.VideoCapture(video_path)
    #     if not cap.isOpened():
    #         print(f"Error opening video: {video_path}")
    #         return np.zeros((config.NUM_FRAMES, config.FRAME_SIZE[0], config.FRAME_SIZE[1], 3), dtype=np.uint8)

    #     # Get metadata
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     if fps <= 0: fps = 25.0
        
    #     # Determine which frame indices we need
    #     duration = total_frames_in_video / fps
    #     timestamps = np.linspace(0, max(0, duration - 0.5), config.NUM_FRAMES)
    #     target_indices = [int(t * fps) for t in timestamps]
        
    #     # Optimize: Sort and remove duplicates to read in order
    #     target_indices = sorted(list(set(target_indices)))
        
    #     frames = []
    #     current_idx = 0
        
    #     # --- SEQUENTIAL READ (No Seeking Errors) ---
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret: 
    #             break # End of video
            
    #         # If this is a frame we want, keep it
    #         if current_idx in target_indices:
    #             frame = cv2.resize(frame, (config.FRAME_SIZE[1], config.FRAME_SIZE[0]))
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             frames.append(frame)
            
    #         current_idx += 1
            
    #         # Optimization: Stop reading if we passed the last frame we need
    #         if target_indices and current_idx > target_indices[-1]:
    #             break

    #     cap.release()

    #     # Handle edge cases (padding)
    #     if len(frames) == 0:
    #          return np.zeros((config.NUM_FRAMES, config.FRAME_SIZE[0], config.FRAME_SIZE[1], 3), dtype=np.uint8)

    #     # If we missed some frames (due to rounding or bad metadata), duplicate the last one
    #     while len(frames) < config.NUM_FRAMES:
    #         frames.append(frames[-1])
            
    #     # If we got too many (due to duplicate indices logic), trim
    #     return np.array(frames[:config.NUM_FRAMES], dtype=np.uint8)
    def _sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return np.zeros((config.NUM_FRAMES, config.FRAME_SIZE[0], config.FRAME_SIZE[1], 3), dtype=np.uint8)

        # Get metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 25.0
        
        # Calculate exact timestamps needed
        duration = total_frames_in_video / fps
        timestamps = np.linspace(0, max(0, duration - 0.5), config.NUM_FRAMES)
        
        # Convert timestamps to frame indices and sort them
        target_indices = sorted(list(set([int(t * fps) for t in timestamps])))
        
        frames = []
        current_idx = 0
        
        # Fast Sequential Read
        while True:
            ret, frame = cap.read()
            if not ret: 
                break # End of video
            
            # Only process if this is a frame we need
            if current_idx in target_indices:
                # Resize and Color Convert (Standard preprocessing)
                frame = cv2.resize(frame, (config.FRAME_SIZE[1], config.FRAME_SIZE[0]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            current_idx += 1
            
            # Optimization: Stop reading if we passed the last frame we need
            if target_indices and current_idx > target_indices[-1]:
                break

        cap.release()

        # Handle edge cases (padding)
        if len(frames) == 0:
             return np.zeros((config.NUM_FRAMES, config.FRAME_SIZE[0], config.FRAME_SIZE[1], 3), dtype=np.uint8)

        # Duplicate last frame if we are missing any (due to video length mismatches)
        while len(frames) < config.NUM_FRAMES:
            frames.append(frames[-1])
            
        return np.array(frames[:config.NUM_FRAMES], dtype=np.uint8)

    def _compute_features_from_landmarks(self, landmarks, image_shape):
        h, w = image_shape[:2]
        points = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks])
        features = []
        
        # 1. Action Units (Eyes)
        features.extend([
            self._eye_aspect_ratio(points, 'left'), self._eye_aspect_ratio(points, 'right'),
            np.linalg.norm(points[33] - points[133]), np.linalg.norm(points[362] - points[263]), # widths
            np.linalg.norm(points[33] - points[263]), # distance
            np.linalg.norm(points[159] - points[145]), np.linalg.norm(points[386] - points[374]), # upper lid
            np.linalg.norm(points[145] - points[153]), np.linalg.norm(points[374] - points[380]), # lower lid
            np.linalg.norm(points[159] - points[153]), np.linalg.norm(points[386] - points[380]), # squint
            abs(np.linalg.norm(points[33] - points[133]) - np.linalg.norm(points[362] - points[263])) # symmetry
        ])
        
        # 2. Eyebrows
        features.extend([
            self._eyebrow_height(points, 'left'), self._eyebrow_height(points, 'right'),
            np.linalg.norm(points[70] - points[27]), np.linalg.norm(points[300] - points[27]),
            np.linalg.norm(points[105] - points[33]), np.linalg.norm(points[334] - points[263]),
            np.linalg.norm(points[70] - points[300]),
            np.arctan2(points[300][1] - points[70][1], points[300][0] - points[70][0])
        ])

        # 3. Mouth
        mouth_width = np.linalg.norm(points[61] - points[291])
        features.extend([
            self._mouth_aspect_ratio(points),
            mouth_width, np.linalg.norm(points[13] - points[14]), # height
            np.linalg.norm(points[0] - points[13]), np.linalg.norm(points[17] - points[14]), # lips
            np.linalg.norm(points[61] - points[291]), np.linalg.norm(points[291] - points[61]), # corners
            np.linalg.norm(points[13] - points[14]), np.linalg.norm(points[0] - points[17]), # open/dist
            abs(np.linalg.norm(points[61] - points[0]) - np.linalg.norm(points[291] - points[0])) / (mouth_width + 1e-6)
        ])

        # 4. Nose/Cheeks
        features.extend([
            np.linalg.norm(points[129] - points[358]), np.linalg.norm(points[1] - points[2]),
            np.linalg.norm(points[206] - points[61]), np.linalg.norm(points[426] - points[291]),
            np.linalg.norm(points[1] - points[152])
        ])

        # 5. Gaze & Pose
        pitch, yaw, roll = self._head_pose(points, w, h)
        left_eye = (points[33] + points[133]) / 2
        right_eye = (points[362] + points[263]) / 2
        features.extend([
            pitch, yaw, roll,
            (left_eye[0] - points[168][0])/w, (left_eye[1] - points[168][1])/h,
            (right_eye[0] - points[168][0])/w, (right_eye[1] - points[168][1])/h,
            abs(((left_eye[0] - points[168][0])/w) - ((right_eye[0] - points[168][0])/w))
        ])

        # 6. Expression/Valence/Arousal
        features.extend([
            self._facial_symmetry(points),
            np.linalg.norm(points[234] - points[454]), np.linalg.norm(points[10] - points[152]),
            (self._eyebrow_height(points, 'left') + self._eyebrow_height(points, 'right'))/2,
            self._mouth_aspect_ratio(points),
            ((np.linalg.norm(points[61] - points[0]) + np.linalg.norm(points[291] - points[0]))/2) / (h + 1e-6),
            (self._eye_aspect_ratio(points, 'left') + self._eye_aspect_ratio(points, 'right') + self._mouth_aspect_ratio(points))/3.0
        ])

        feat_arr = np.array(features, dtype=np.float32)
        # Pad or clip to ensure exactly 50 features
        if len(feat_arr) > 50: feat_arr = feat_arr[:50]
        elif len(feat_arr) < 50: feat_arr = np.pad(feat_arr, (0, 50-len(feat_arr)))
            
        # Normalize
        mean, std = feat_arr.mean(), feat_arr.std()
        if std > 1e-6: feat_arr = (feat_arr - mean) / std
        return feat_arr

    def _eye_aspect_ratio(self, points, side):
        if side == 'left': p = [33, 160, 158, 133, 153, 144]
        else: p = [362, 385, 387, 263, 373, 380]
        v = np.linalg.norm(points[p[1]] - points[p[5]]) + np.linalg.norm(points[p[2]] - points[p[4]])
        h = np.linalg.norm(points[p[0]] - points[p[3]])
        return v / (2.0 * h + 1e-6)

    def _eyebrow_height(self, points, side):
        bp, ep = (70, 33) if side == 'left' else (300, 263)
        return np.linalg.norm(points[bp] - points[ep])

    def _mouth_aspect_ratio(self, points):
        return np.linalg.norm(points[13] - points[14]) / (np.linalg.norm(points[61] - points[291]) + 1e-6)

    def _head_pose(self, points, w, h):
        nose, chin = points[1], points[152]
        leye, reye = points[33], points[263]
        yaw = np.arctan2(nose[0] - (leye+reye)[0]/2, nose[2] - (leye+reye)[2]/2 + 1e-6)
        pitch = np.arctan2(nose[1] - chin[1], abs(nose[2] - chin[2]) + 1e-6)
        roll = np.arctan2(reye[1] - leye[1], reye[0] - leye[0] + 1e-6)
        return pitch, yaw, roll

    def _facial_symmetry(self, points):
        l_pts, r_pts = [33, 133, 61, 206], [263, 362, 291, 426]
        scores = []
        cx = points[1][0]
        for l, r in zip(l_pts, r_pts):
            ld, rd = abs(points[l][0] - cx), abs(points[r][0] - cx)
            scores.append(1.0 - abs(ld - rd)/(ld + rd + 1e-6))
        return np.mean(scores)