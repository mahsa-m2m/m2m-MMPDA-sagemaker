import json
import boto3
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import tempfile
import os
from typing import Dict, Any


class AudioPreprocessingError(Exception):
    pass

class S3WriteError(Exception):
    pass

def parse_s3_path(s3_path: str) -> tuple:
    parts = s3_path.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


class AudioFeatureExtractor:
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
        self.target_sr = 16000
    
    def _load_audio_from_file(self, audio_path: str) -> np.ndarray:
        """Load audio from regular audio file (.wav, .mp3, etc.)"""
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return audio
    
    def _load_audio_from_pt(self, pt_path: str) -> np.ndarray:
        """
        Load audio from .pt file created by video preprocessing.
        
        Expected .pt file format:
        {
            'vision_behaviour': tensor,
            'vision_face': tensor,
            'audio_mel': tensor,        # NOT used by audio inference
            'audio_wave': tensor        # USED - raw waveform for Wav2Vec2
        }
        """
        try:
            # Load the .pt file
            sample = torch.load(pt_path, map_location='cpu')
            
            # Extract audio_wave (raw waveform)
            if 'audio_wave' not in sample:
                raise AudioPreprocessingError(
                    f".pt file missing 'audio_wave' key. Available keys: {list(sample.keys())}"
                )
            
            audio_wave = sample['audio_wave']
            
            # Convert to numpy
            if isinstance(audio_wave, torch.Tensor):
                audio = audio_wave.numpy()
            else:
                audio = np.array(audio_wave)
            
            # Ensure it's 1D (mono)
            if audio.ndim > 1:
                audio = audio.squeeze()
            
            return audio
            
        except Exception as e:
            raise AudioPreprocessingError(f"Failed to load .pt file: {str(e)}")
    
    def _load_audio_from_tensor(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """Load audio from PyTorch tensor directly"""
        audio = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else np.array(audio_tensor)
        
        # Ensure it's 1D (mono)
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        return audio

    def extract(self, audio_input) -> np.ndarray:
        """
        Extract Wav2Vec2 features from audio input.
        
        Supports three input types:
        1. File path to audio file (.wav, .mp3, etc.) - for audio-only pipeline
        2. File path to .pt file - for fusion pipeline from video preprocessing
        3. PyTorch tensor directly - for in-memory processing
        
        Args:
            audio_input: File path (str) or PyTorch tensor
            
        Returns:
            np.ndarray: Wav2Vec2 embeddings (768-dimensional)
        """
        
        # Determine input type and load audio
        if isinstance(audio_input, str):
            # It's a file path
            if audio_input.endswith('.pt'):
                # Load from .pt file (fusion pipeline)
                audio = self._load_audio_from_pt(audio_input)
            else:
                # Load from regular audio file (audio-only pipeline)
                audio = self._load_audio_from_file(audio_input)
        
        elif isinstance(audio_input, torch.Tensor):
            # Direct tensor input
            audio = self._load_audio_from_tensor(audio_input)
        
        else:
            raise AudioPreprocessingError(
                f"Unsupported input type: {type(audio_input)}. "
                f"Expected str (file path) or torch.Tensor"
            )
        
        inputs = self.processor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return features

feature_extractor = None


def get_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        feature_extractor = AudioFeatureExtractor()
    return feature_extractor


def lambda_handler(event: Dict[str, Any], context=None) -> Dict[str, Any]:
    s3_client = boto3.client('s3')
    
    try:
        session_id = event['sessionId']
        file_type = event['fileType']
        chunk_id = event['chunkId']
        s3_input = event['s3Input']
        metadata = event.get('metadata', {})
        
        input_bucket, input_key = parse_s3_path(s3_input)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            s3_client.download_file(input_bucket, input_key, tmp_file.name)
            chunk_path = tmp_file.name
        
        try:
            extractor = get_feature_extractor()
            features = extractor.extract(chunk_path)
            
            embeddings_data = {
                "features": features.tolist(),
                "shape": list(features.shape),
                "dtype": str(features.dtype)
            }
            
            output_key = f"results/{session_id}/audio/{chunk_id}/embeddings.json"
            s3_output = f"s3://{input_bucket}/{output_key}"
            
            try:
                s3_client.put_object(
                    Bucket=input_bucket,
                    Key=output_key,
                    Body=json.dumps(embeddings_data),
                    ContentType='application/json'
                )
            except Exception as e:
                raise S3WriteError(f"Failed to write embeddings: {str(e)}")
            
        finally:
            os.unlink(chunk_path)
        
        return {
            "sessionId": session_id,
            "fileType": file_type,
            "chunkId": chunk_id,
            "s3Output": s3_output,
            "status": "success",
            "metadata": metadata,
            "error": None
        }
        
    except KeyError as e:
        return {
            "sessionId": event.get('sessionId', 'unknown'),
            "fileType": event.get('fileType', 'unknown'),
            "chunkId": event.get('chunkId', 'unknown'),
            "s3Output": "",
            "status": "failed",
            "metadata": {},
            "error": f"Missing required field: {str(e)}"
        }
    except S3WriteError as e:
        return {
            "sessionId": event.get('sessionId', 'unknown'),
            "fileType": event.get('fileType', 'unknown'),
            "chunkId": event.get('chunkId', 'unknown'),
            "s3Output": "",
            "status": "failed",
            "metadata": {},
            "error": str(e)
        }
    except Exception as e:
        return {
            "sessionId": event.get('sessionId', 'unknown'),
            "fileType": event.get('fileType', 'unknown'),
            "chunkId": event.get('chunkId', 'unknown'),
            "s3Output": "",
            "status": "failed",
            "metadata": {},
            "error": f"AudioPreprocessingError: {str(e)}"
        }

