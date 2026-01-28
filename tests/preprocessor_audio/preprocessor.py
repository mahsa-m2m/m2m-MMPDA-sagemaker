#!/usr/bin/env python3
import os
import json
import sys
import boto3
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import tempfile
import traceback


class AudioPreprocessingError(Exception):
    pass

class S3WriteError(Exception):
    pass

def parse_s3_path(s3_path: str) -> tuple:
    parts = s3_path.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]

def download_s3_folder_preserve_structure(s3_client, bucket, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            
            relative_path = key[len(prefix):].lstrip('/')
            if not relative_path:
                continue
                
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3_client.download_file(bucket, key, local_file_path)


class AudioFeatureExtractor:
    def __init__(self, model_s3_path: str = None):
        s3_client = boto3.client('s3')
        
        # Target sampling rate
        self.target_sr = 16000

        if model_s3_path:
            # Load processor and model from S3
            try:
                model_bucket, model_prefix = parse_s3_path(model_s3_path)
                model_dir = tempfile.mkdtemp()
                
                print(f"Downloading model from S3: {model_s3_path}")
                download_s3_folder_preserve_structure(
                    s3_client, model_bucket, model_prefix, model_dir
                )
                self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
                self.model = Wav2Vec2Model.from_pretrained(model_dir)
                print("Processor and model loaded successfully from S3")
            except Exception as e:
                error_trace = traceback.format_exc()
                print(f"Failed to load model from S3: {str(e)}")
                print(f"Full traceback:\n{error_trace}")
                raise AudioPreprocessingError(f"Failed to load model from S3: {str(e)}")
        else:
            # Load processor and model from HuggingFace
            print("Loading processor and model from HuggingFace (no S3 path provided)")
            model_name = "facebook/wav2vec2-base-960h"
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            print("Processor and model loaded successfully from HuggingFace")
        
        self.model.eval()

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
            'audio_mel': tensor,        # NOT used
            'audio_wave': tensor        # USED - raw waveform for Wav2Vec2
        }
        """
        try:
            # Load the .pt file
            sample = torch.load(pt_path, map_location='cpu')
            
            # Extract audio_wave
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
    """
    Returns an instance of AudioFeatureExtractor.
    """
    global feature_extractor
    if feature_extractor is None:
        feature_extractor = AudioFeatureExtractor()
    return feature_extractor

def main():
    s3_client = boto3.client('s3')
    
    session_id = os.environ.get("SESSION_ID", "unknown")
    chunk_id = os.environ.get("CHUNK_ID", "unknown")
    s3_input = os.environ.get("S3_INPUT", "")
    model_s3_path = os.environ.get("MODEL_S3_PATH", "")
    
    print(f"Environment variables:")
    print(f"  SESSION_ID: {session_id}")
    print(f"  CHUNK_ID: {chunk_id}")
    print(f"  S3_INPUT: {s3_input}")
    print(f"  MODEL_S3_PATH: {model_s3_path}")
    
    try:
        if not s3_input:
            raise AudioPreprocessingError("Missing S3_INPUT")
        
        input_bucket, input_key = parse_s3_path(s3_input)
        
        # Determine file extension to preserve it locally
        _, file_extension = os.path.splitext(input_key)
        if not file_extension:
            file_extension = '.wav' # Default 
            
        print(f"Downloading file from S3: s3://{input_bucket}/{input_key} with extension {file_extension}")
        
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
            s3_client.download_file(input_bucket, input_key, tmp_file.name)
            chunk_path = tmp_file.name
        
        try:
            print("Initializing AudioFeatureExtractor...")
            extractor = AudioFeatureExtractor(
                model_s3_path if model_s3_path else None
            )
            print("Extracting features from input...")
            features = extractor.extract(chunk_path)
            print(f"Features extracted successfully: shape={features.shape}")
            
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
            if os.path.exists(chunk_path):
                os.unlink(chunk_path)
        
        output_data = {
            "sessionId": session_id,
            "fileType": "audio",
            "chunkId": chunk_id,
            "s3Output": s3_output,
            "status": "success",
            "metadata": {},
            "error": None
        }
        print(json.dumps(output_data))
        sys.exit(0)
        
    except AudioPreprocessingError as e:
        error_trace = traceback.format_exc()
        print(f"AudioPreprocessingError: {str(e)}", file=sys.stderr)
        print(f"Full traceback:\n{error_trace}", file=sys.stderr)
        output_data = {
            "sessionId": session_id,
            "fileType": "audio",
            "chunkId": chunk_id,
            "s3Output": "",
            "status": "failed",
            "metadata": {},
            "error": str(e),
            "traceback": error_trace
        }
        print(json.dumps(output_data), file=sys.stderr)
        sys.exit(1)
    except S3WriteError as e:
        error_trace = traceback.format_exc()
        print(f"S3WriteError: {str(e)}", file=sys.stderr)
        print(f"Full traceback:\n{error_trace}", file=sys.stderr)
        output_data = {
            "sessionId": session_id,
            "fileType": "audio",
            "chunkId": chunk_id,
            "s3Output": "",
            "status": "failed",
            "metadata": {},
            "error": str(e),
            "traceback": error_trace
        }
        print(json.dumps(output_data), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Unexpected Exception [{type(e).__name__}]: {str(e)}", file=sys.stderr)
        print(f"Full traceback:\n{error_trace}", file=sys.stderr)
        output_data = {
            "sessionId": session_id,
            "fileType": "audio",
            "chunkId": chunk_id,
            "s3Output": "",
            "status": "failed",
            "metadata": {},
            "error": f"AudioPreprocessingError: {str(e)}",
            "traceback": error_trace
        }
        print(json.dumps(output_data), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()