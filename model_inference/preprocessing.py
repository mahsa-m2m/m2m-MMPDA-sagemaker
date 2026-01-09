import os
import cv2
import torch
import torchaudio
import numpy as np
import mediapipe as mp
import subprocess
import io
import math
import config


class InferencePreprocessorMMPDA:
    """
    Feature extraction pipeline for MMPDA deception detection.
    Extracts visual (face crops + behavioral features) and audio features.
    """
    
    def __init__(self, model_asset_path="face_landmarker.task"):
        """
        Initialize preprocessing components.
        
        Args:
            model_asset_path: Path to the MediaPipe face_landmarker.task file
        """
        self.model_asset_path = model_asset_path
        self.landmarker = None
        
        # Audio Transforms (matches training code exactly)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_mels=config.N_MELS,
            n_fft=1024,  # Matches training code
            win_length=400,
            hop_length=160
        )
        
        print("✅ InferencePreprocessorMMPDA initialized")

    def _init_mediapipe(self):
        """
        Initialize MediaPipe FaceLandmarker (lazy initialization).
        Uses the new MediaPipe Tasks API.
        """
        if self.landmarker is None:
            base_options = mp.tasks.BaseOptions(model_asset_path=self.model_asset_path)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,  # For behavioral features
                output_facial_transformation_matrixes=True,  # For pose estimation
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # ==========================================
    # MAIN PROCESSING ENTRY POINT
    # ==========================================
    
    def process_video(self, video_path):
        """
        Main entry point: Extracts all features from video.
        
        Args:
            video_path (str): Path to input video file
            
        Returns:
            dict: Dictionary containing:
                - 'vision_behaviour': [1, NUM_FRAMES, 50] - Behavioral features
                - 'vision_face': [1, 3, NUM_FRAMES, H, W] - Face crops (normalized [0,1])
                - 'audio_mel': [1, 3, N_MELS, T] - Mel spectrogram - <<None for the video model>>
                - 'audio_wave': [1, AUDIO_LENGTH] - Raw audio waveform - <<None for the video model>>
                
        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        # Initialize MediaPipe
        self._init_mediapipe()

        # Validate input
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1. Sample Frames (NumPy array: [NUM_FRAMES, H, W, 3])
        raw_frames_np = self._sample_frames(video_path) 
        
        # 2. Extract Features AND Crops
        behavioral_np, face_crops_np = self._extract_features_and_crops(raw_frames_np)

        # 3. Extract Audio
        # audio_wave_np, audio_mel_np = self._extract_audio(video_path)

        # 4. Prepare Tensors
        # Vision Face: [N, H, W, 3] -> [3, N, H, W] -> Normalize to [0, 1]
        vision_face = torch.from_numpy(face_crops_np).permute(3, 0, 1, 2).float()
        vision_face = vision_face / 255.0  # Normalize from uint8 to [0, 1]

        # Vision Behaviour: [N, 50]
        vision_behaviour = torch.from_numpy(behavioral_np).float()
        
        # # Audio Wave: [AUDIO_LENGTH]
        # audio_wave = torch.from_numpy(audio_wave_np).float()
        audio_wave = torch.zeros(config.AUDIO_LENGTH, dtype=torch.float32) # NONE for video model
        
        # # Audio Mel: [3, N_MELS, T]
        # audio_mel = torch.from_numpy(audio_mel_np).float()
        T = (config.AUDIO_LENGTH // 160) + 1
        audio_mel = torch.zeros(3, config.N_MELS, T, dtype=torch.float32) # NONE for video model

        # 5. Add Batch Dimension (B=1) - Ready for model input
        return {
            'vision_behaviour': vision_behaviour.unsqueeze(0),  # [1, N, 50]
            'vision_face': vision_face.unsqueeze(0),            # [1, 3, N, H, W]
            'audio_mel': audio_mel.unsqueeze(0),                # [1, 3, N_MELS, T]
            'audio_wave': audio_wave.unsqueeze(0)               # [1, AUDIO_LENGTH]
        }

    # ==========================================
    # FEATURE EXTRACTION
    # ==========================================
    
    def _extract_features_and_crops(self, frames):
        """
        Extract blendshape features and face crops from frames.
        
        Args:
            frames (np.ndarray): Array of shape [NUM_FRAMES, H, W, 3] (RGB, uint8)
            
        Returns:
            tuple: (behavioral_features, face_crops)
                - behavioral_features: [NUM_FRAMES, 50] (float32)
                - face_crops: [NUM_FRAMES, H, W, 3] (uint8)
        """
        final_features = []
        final_crops = []

        for frame in frames:
            # MediaPipe expects RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection = self.landmarker.detect(mp_image)
            
            # Default empty features
            feat_vec = np.zeros(50, dtype=np.float32)
            
            # Default crop is the resized full frame (fallback)
            face_crop = cv2.resize(frame, (config.FRAME_SIZE[1], config.FRAME_SIZE[0]))
            
            if detection.face_landmarks:
                # 1. Extract 50-dimensional behavioral features
                feat_vec = self._get_mediapipe_features_logic(
                    detection.face_blendshapes[0], 
                    detection.facial_transformation_matrixes[0]
                )
                
                # 2. Extract Face Crop with 20% margin
                landmarks = detection.face_landmarks[0]
                h, w = frame.shape[:2]
                x_c = [lm.x for lm in landmarks]
                y_c = [lm.y for lm in landmarks]
                x_min, x_max = min(x_c) * w, max(x_c) * w
                y_min, y_max = min(y_c) * h, max(y_c) * h
                
                # Apply 20% margin
                margin_x, margin_y = (x_max - x_min) * 0.2, (y_max - y_min) * 0.2
                x1, y1 = max(0, int(x_min - margin_x)), max(0, int(y_min - margin_y))
                x2, y2 = min(w, int(x_max + margin_x)), min(h, int(y_max + margin_y))
                
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    face_crop = cv2.resize(crop, (config.FRAME_SIZE[1], config.FRAME_SIZE[0]))

            final_features.append(feat_vec)
            final_crops.append(face_crop)
            
        return np.array(final_features, dtype=np.float32), np.array(final_crops, dtype=np.uint8)

    def _get_mediapipe_features_logic(self, blendshapes, matrix):
        """
        Maps MediaPipe output to 50-dimensional feature vector.
        
        Features breakdown:
        - Action Units (35 dim): Facial muscle movements
        - Gaze (8 dim): Head pose (pitch, yaw, roll) + eye gaze directions
        - Emotions (5 dim): Happy, sad, surprise, fear, anger
        - Valence/Arousal (2 dim): Emotional dimensions
        
        Args:
            blendshapes: MediaPipe face blendshapes
            matrix: MediaPipe facial transformation matrix (4x4)
            
        Returns:
            np.ndarray: 50-dimensional feature vector (float32)
        """
        # Convert blendshapes to dictionary for easy access
        bs = {b.category_name: b.score for b in blendshapes}
        au = np.zeros(35, dtype=np.float32)
        
        # ==========================================
        # ACTION UNITS MAPPING (35 features)
        # ==========================================
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
        
        # Additional action units
        extras = ['mouthFunnel', 'mouthPressLeft', 'mouthPressRight', 'mouthRollUpper', 
                  'mouthRollLower', 'mouthShrugUpper', 'jawLeft', 'jawRight', 'jawForward',
                  'cheekPuff', 'mouthLowerDownLeft', 'mouthLowerDownRight']
        for i, name in enumerate(extras):
            if 17 + i < 35:
                au[17 + i] = bs.get(name, 0)

        # ==========================================
        # HEAD POSE (3 features: pitch, yaw, roll)
        # ==========================================
        m = np.array(matrix)
        if m.shape == (4, 4):
            m = m.flatten()
        
        sy = math.sqrt(m[0] * m[0] + m[4] * m[4])
        if sy > 1e-6:
            pitch = math.atan2(m[9], m[10])
            yaw = math.atan2(-m[8], sy)
            roll = math.atan2(m[4], m[0])
        else:
            pitch = math.atan2(-m[6], m[5])
            yaw = math.atan2(-m[8], sy)
            roll = 0

        # ==========================================
        # GAZE (5 features + asymmetry)
        # ==========================================
        l_dx = bs.get('eyeLookInLeft', 0) - bs.get('eyeLookOutLeft', 0)
        l_dy = bs.get('eyeLookUpLeft', 0) - bs.get('eyeLookDownLeft', 0)
        r_dx = bs.get('eyeLookOutRight', 0) - bs.get('eyeLookInRight', 0)
        r_dy = bs.get('eyeLookUpRight', 0) - bs.get('eyeLookDownRight', 0)
        gaze = np.array([pitch, yaw, roll, l_dx, l_dy, r_dx, r_dy, abs(l_dx - r_dx)], 
                       dtype=np.float32)

        # ==========================================
        # EMOTIONS (5 features)
        # ==========================================
        happy = bs.get('mouthSmileLeft', 0) * 0.5 + bs.get('mouthSmileRight', 0) * 0.5
        sad = (bs.get('mouthFrownLeft', 0) + bs.get('browDownLeft', 0)) / 2.0
        surprise = (bs.get('browInnerUp', 0) + bs.get('jawOpen', 0)) / 2.0
        fear = (bs.get('mouthStretchLeft', 0) + bs.get('browInnerUp', 0)) / 2.0
        anger = (bs.get('browDownLeft', 0) + bs.get('jawForward', 0)) / 2.0
        emotions = np.array([happy, sad, surprise, fear, anger], dtype=np.float32)

        # ==========================================
        # VALENCE/AROUSAL (2 features)
        # ==========================================
        val = happy - max(sad, anger, fear)
        arousal = max(happy, surprise, anger, fear)
        va = np.array([val, arousal], dtype=np.float32)

        # Concatenate all features: 35 + 8 + 5 + 2 = 50
        return np.concatenate([au, gaze, emotions, va])

    # ==========================================
    # AUDIO EXTRACTION
    # ==========================================
    
    def _extract_audio(self, video_path):
        """
        Extracts audio directly to memory (no disk I/O).
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            tuple: (waveform, mel_spectrogram) as numpy arrays
                - waveform: [AUDIO_LENGTH] float32
                - mel_spectrogram: [3, N_MELS, T] float32
        """
        try:
            # FFmpeg command to pipe audio as WAV to stdout
            cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-f', 'wav',
                '-acodec', 'pcm_s16le', '-ar', str(config.SAMPLE_RATE), 
                '-ac', '1', '-loglevel', 'error', 'pipe:1'
            ]
            
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                check=True
            )
            
            # Read from memory buffer
            memory_file = io.BytesIO(process.stdout)
            waveform, sr = torchaudio.load(memory_file)

            if waveform.shape[1] == 0:
                raise ValueError("Empty audio")

            # Pad or trim to target length
            if waveform.shape[1] < config.AUDIO_LENGTH:
                waveform = torch.nn.functional.pad(
                    waveform, 
                    (0, config.AUDIO_LENGTH - waveform.shape[1])
                )
            else:
                waveform = waveform[:, :config.AUDIO_LENGTH]

            # Generate mel spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec = mel_spec.repeat(3, 1, 1)  # Convert to 3 channels

            return waveform.squeeze(0).numpy(), mel_spec.numpy()
            
        except Exception as e:
            # Fallback to silent audio
            print(f"⚠️  Audio extraction failed: {e}. Using silent audio.")
            return self._get_silent_audio()

    def _get_silent_audio(self):
        """Generate silent audio fallback"""
        silent_wave = np.zeros(config.AUDIO_LENGTH, dtype=np.float32)
        n_time_steps = (config.AUDIO_LENGTH // 160) + 1
        silent_mel = np.zeros((3, config.N_MELS, n_time_steps), dtype=np.float32)
        return silent_wave, silent_mel

    # ==========================================
    # FRAME SAMPLING
    # ==========================================
    
    def _sample_frames(self, video_path):
        """
        Samples frames uniformly from video.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            np.ndarray: Array of shape [NUM_FRAMES, H, W, 3] (uint8, RGB)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️  Error opening video: {video_path}")
            return np.zeros(
                (config.NUM_FRAMES, config.FRAME_SIZE[0], config.FRAME_SIZE[1], 3), 
                dtype=np.uint8
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if total_frames > 0:
            # Calculate frame indices to sample
            indices = np.linspace(0, total_frames - 1, config.NUM_FRAMES).astype(int)
            last_valid = None
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    last_valid = frame.copy()
                elif last_valid is not None:
                    # Use last valid frame if current read fails
                    frames.append(last_valid.copy())
        
        cap.release()
        
        # Handle edge cases
        if len(frames) > 0:
            # Pad with last frame if needed
            while len(frames) < config.NUM_FRAMES:
                frames.append(frames[-1].copy())
            return np.array(frames[:config.NUM_FRAMES], dtype=np.uint8)
        
        # Return zeros if no frames could be read
        return np.zeros(
            (config.NUM_FRAMES, config.FRAME_SIZE[0], config.FRAME_SIZE[1], 3), 
            dtype=np.uint8
        )

    # ==========================================
    # CLEANUP
    # ==========================================
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'landmarker') and self.landmarker is not None:
            self.landmarker.close()


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print(f"Testing MMPDA preprocessing on: {video_path}")
    
    # Initialize preprocessor
    preprocessor = InferencePreprocessorMMPDA()
    
    # Process video
    try:
        features = preprocessor.process_video(video_path)
        
        print("\nFeature extraction successful!")
        print("\nExtracted features:")
        for key, tensor in features.items():
            print(f"  {key:20s}: {tensor.shape}")
        
        # Verify shapes
        assert features['vision_behaviour'].shape[0] == 1, "Batch dimension missing"
        assert features['vision_behaviour'].shape[2] == 50, "Expected 50 behavioral features"
        assert features['vision_face'].shape[0] == 1, "Batch dimension missing"
        assert features['vision_face'].shape[1] == 3, "Expected 3 channels"
        assert features['audio_mel'].shape[0] == 1, "Batch dimension missing"
        assert features['audio_wave'].shape[0] == 1, "Batch dimension missing"
        
        print("\nAll shape validations passed!")
        print(f"\nFeature details:")
        print(f"  Behavioral features: {features['vision_behaviour'].shape[1]} frames x 50 dims")
        print(f"  Face crops: {features['vision_face'].shape[2]} frames x {features['vision_face'].shape[3]}x{features['vision_face'].shape[4]}")
        print(f"  Audio mel: {features['audio_mel'].shape[3]} time steps")
        print(f"  Audio wave: {features['audio_wave'].shape[1]} samples")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()