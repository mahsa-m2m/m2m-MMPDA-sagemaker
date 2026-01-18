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
    
    def extract(self, audio_path: str) -> np.ndarray:
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
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