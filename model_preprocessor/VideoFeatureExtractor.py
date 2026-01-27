import os
import torch
import subprocess
import numpy as np
import cv2
import math
import torchaudio
import mediapipe as mp
import logging
import tempfile
import config 

from utils import VideoPreprocessError, AudioExtractionError, InvalidInputError

logger = logging.getLogger(__name__)

class VideoFeatureExtractor:
    def __init__(self):
        """
        Initializes the preprocessor.
        """
        logger.info("Initializing VideoFeatureExtractor")

        # 1. Load from Config
        self.num_frames = config.NUM_FRAMES
        self.frame_size = config.FRAME_SIZE
        self.audio_length = config.AUDIO_LENGTH
        self.sample_rate = config.SAMPLE_RATE
        self.model_path = config.MODEL_ASSET

        # 2. MediaPipe Model Exists
        self._ensure_model_exists()

        logger.info("VideoFeatureExtractor initialized successfully")

    def _ensure_model_exists(self):
        """Checks for the MediaPipe task file."""
        if not os.path.exists(self.model_path):
            logger.info(f"Face Model not found in: {self.model_path}...")
            # url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            # try:
            #     subprocess.check_call(["wget", "-O", self.model_path, url], stdout=subprocess.DEVNULL)
            # except Exception as e:
            #     try:
            #         import requests
            #         response = requests.get(url)
            #         with open(self.model_path, 'wb') as f:
            #             f.write(response.content)
            #     except Exception as req_e:
            #         logger.error(f"Failed to download MediaPipe model: {str(req_e)}")
            #         raise VideoPreprocessError(f"Failed to download model: {str(req_e)}")

    def _get_mediapipe_features(self, blendshapes, matrix):
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

    def _sample_frames(self, video_path):
        """Reads video frames using OpenCV."""
        logger.debug(f"Sampling frames from {video_path}")
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
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
                while len(frames) < self.num_frames:
                    frames.append(frames[-1].copy())
                return np.array(frames[:self.num_frames], dtype=np.uint8)
            
            raise VideoPreprocessError("Video contains no valid frames.")
        except Exception as e:
            logger.error(f"Frame sampling failed: {str(e)}")
            raise VideoPreprocessError(f"OpenCV frame sampling failed: {str(e)}")

    def _extract_audio(self, video_path, n_mels=128):
        """Robust audio extraction."""
        logger.debug("Extracting audio from video")
        target_length = int(config.AUDIO_DURATION * self.sample_rate)

        try:
            # 1. Direct Load
            waveform, sr = torchaudio.load(video_path)
        except Exception:
            # 2. FFmpeg Fallback
            temp_wav = None
            try:
                fd, temp_wav = tempfile.mkstemp(suffix='.wav')
                os.close(fd) 
                
                # FFmpeg command:
                # -y: overwrite
                # -i: input video
                # -vn: disable video
                # -acodec pcm_s16le: standard wav encoding
                # -ar: force sample rate (avoids resampling later)
                # -ac: force channels (1 for mono) to simplify
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vn", 
                    "-acodec", "pcm_s16le", 
                    "-ar", str(self.sample_rate),
                    "-ac", "1",
                    "-t", str(config.AUDIO_DURATION),
                    temp_wav
                ]
                
                # Run FFmpeg silently
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Load the clean WAV file
                waveform, sr = torchaudio.load(temp_wav)
                
            except Exception as ffmpeg_e:
                 raise AudioExtractionError(f"Audio extraction failed (Direct & FFmpeg): {str(ffmpeg_e)}")
            
            finally:
                # Cleanup temp wav file
                if temp_wav and os.path.exists(temp_wav):
                    os.remove(temp_wav)

        try:
            # Resample
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # # Pad/Trim
            # if waveform.shape[1] < self.audio_length:
            #     waveform = torch.nn.functional.pad(waveform, (0, self.audio_length - waveform.shape[1]))
            # else:
            #     waveform = waveform[:, :self.audio_length]

            # Pad/Trim
            if waveform.shape[1] < target_length:
                waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
            else:
                waveform = waveform[:, target_length]

            # Mel Spec
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_mels=n_mels, n_fft=1024, win_length=400, hop_length=160
            )
            mel_spec = mel_transform(waveform)
            mel_spec = mel_spec.repeat(3, 1, 1)

            return waveform.squeeze(0).numpy(), mel_spec.numpy()
        
        except Exception as e:
            raise AudioExtractionError(f"Audio tensor processing failed: {str(e)}")

    def _get_zero_audio(self, n_mels=128):
        """Generates zero-filled audio variables with exact shapes."""
        
        # Shape: (1, audio_length)
        zero_waveform = torch.zeros((1, self.audio_length))
        
        # To get matching dimensions
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, 
            n_mels=n_mels, 
            n_fft=1024, 
            win_length=400, 
            hop_length=160
        )
        
        # Generate Mel Spec
        zero_mel_spec = mel_transform(zero_waveform)
        
        # Match Channel Repetition (3 channels)
        zero_mel_spec = zero_mel_spec.repeat(3, 1, 1)
        
        # Return as Numpy (to match _extract_audio output signature)
        return zero_waveform.squeeze(0).numpy(), zero_mel_spec.numpy()

    def process_video(self, video_path, extract_audio=False):
        """
        Main method
        
        Args:
            video_path (str): Path to the video file.
            extract_audio (bool): If False, fills audio with zeros directly.
            
        Returns:
            dict: Dictionary containing tensors ready for the model, or None on failure.
        """
        logger.info(f"Starting video processing for: {video_path}")
        if not os.path.exists(video_path):
            raise InvalidInputError(f"Local video file not found at {video_path}")

        try:
            # 1. Sample Frames
            frames = self._sample_frames(video_path)
            # if frames is None:
            #     print("❌ Error: Could not sample frames.")
            #     return None

            # 2. Setup MediaPipe
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            base_options = BaseOptions(model_asset_path=self.model_path)
            options = FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5
            )
            
            final_crops = []
            final_features = []
            
            # Use context manager
            with FaceLandmarker.create_from_options(options) as landmarker:
                for frame in frames:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    detection = landmarker.detect(mp_image)
                    
                    feat_vec = np.zeros(50, dtype=np.float32)
                    face_crop = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
                    
                    if detection.face_landmarks:
                        # Extract Features
                        feat_vec = self._get_mediapipe_features(
                            detection.face_blendshapes[0], 
                            detection.facial_transformation_matrixes[0]
                        )
                        
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
                            face_crop = cv2.resize(crop, (self.frame_size[1], self.frame_size[0]))

                    final_features.append(feat_vec)
                    final_crops.append(face_crop)
        except Exception as e:
            logger.error(f"MediaPipe processing failed: {str(e)}")
            raise VideoPreprocessError(f"MediaPipe processing failed: {str(e)}")
            
        # 3. Audio
        audio_wave = None
        audio_mel = None
        
        if extract_audio:
            try:
                # extraction
                audio_wave, audio_mel = self._extract_audio(video_path)
            except Exception as e:
                logger.warning(f"Audio extraction failed ({e}). Falling back to zeros.")
                audio_wave, audio_mel = self._get_zero_audio(config.N_MELS)
        else:
            # Skip Audio, generate valid zeros
            audio_wave, audio_mel = self._get_zero_audio(config.N_MELS)

        # 4. Format Tensors
        logger.info("Formatting final tensors")
        frames_tensor = torch.from_numpy(np.array(final_crops)).permute(3, 0, 1, 2).to(torch.uint8)
        feat_tensor = torch.from_numpy(np.array(final_features)).float()
        
        audio_wave_tensor = torch.from_numpy(audio_wave).float()
        audio_mel_tensor = torch.from_numpy(audio_mel).float()
        
        sample = {
            'vision_behaviour': feat_tensor, #.unsqueeze(0), # Add batch dim
            'vision_face': frames_tensor, #.unsqueeze(0),    # Add batch dim
            'audio_mel': audio_mel_tensor, #.unsqueeze(0),
            'audio_wave': audio_wave_tensor #.unsqueeze(0),
            # 'videoname': os.path.basename(video_path)
        }
        
        return sample

