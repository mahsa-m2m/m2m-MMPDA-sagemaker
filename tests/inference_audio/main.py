#!/usr/bin/env python3
import os
import json
import sys
import boto3
import torch
import torch.nn as nn
import joblib
import numpy as np


class AudioInferenceError(Exception):
    pass


class S3WriteError(Exception):
    pass


def parse_s3_path(s3_path: str) -> tuple:
    parts = s3_path.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_heads=8, dropout=0.5):
        super(BiLSTMAttentionModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = attn_out.mean(dim=1)
        x = self.fc(x)
        return x


def load_model_from_s3(s3_client, model_s3_path, scaler_s3_path):
    try:
        model_bucket, model_key = parse_s3_path(model_s3_path)
        scaler_bucket, scaler_key = parse_s3_path(scaler_s3_path)
        
        model_response = s3_client.get_object(Bucket=model_bucket, Key=model_key)
        model_data = model_response['Body'].read()
        
        with open('/tmp/audio_model.pth', 'wb') as f:
            f.write(model_data)
        
        scaler_response = s3_client.get_object(Bucket=scaler_bucket, Key=scaler_key)
        scaler_data = scaler_response['Body'].read()
        
        with open('/tmp/scaler.pkl', 'wb') as f:
            f.write(scaler_data)
        
        state_dict = torch.load('/tmp/audio_model.pth', map_location='cpu')
        model = BiLSTMAttentionModel(
            input_size=768,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            dropout=0.5
        )
        model.load_state_dict(state_dict)
        model.eval()
        
        scaler = joblib.load('/tmp/scaler.pkl')
        
        return model, scaler
        
    except Exception as e:
        raise AudioInferenceError(f"Failed to load model: {str(e)}")


def predict(model, scaler, features):
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    features_scaled = scaler.transform(features)
    X = torch.FloatTensor(features_scaled)
    
    with torch.no_grad():
        output = model(X)
        prob = torch.sigmoid(output).item()
        predicted_class = 1 if prob > 0.5 else 0
        confidence = prob if predicted_class == 1 else 1 - prob
    
    return predicted_class, confidence


def main():
    s3_client = boto3.client('s3')
    
    session_id = os.environ.get("SESSION_ID", "unknown")
    chunk_id = os.environ.get("CHUNK_ID", "unknown")
    s3_input = os.environ.get("S3_INPUT", "")
    model_s3_path = os.environ.get("MODEL_S3_PATH", "")
    scaler_s3_path = os.environ.get("SCALER_S3_PATH", "")
    
    try:
        if not s3_input:
            raise AudioInferenceError("Missing S3_INPUT")
        if not model_s3_path:
            raise AudioInferenceError("Missing MODEL_S3_PATH")
        if not scaler_s3_path:
            raise AudioInferenceError("Missing SCALER_S3_PATH")
        
        model, scaler = load_model_from_s3(s3_client, model_s3_path, scaler_s3_path)
        
        input_bucket, input_key = parse_s3_path(s3_input)
        response = s3_client.get_object(Bucket=input_bucket, Key=input_key)
        embeddings_data = json.loads(response['Body'].read().decode('utf-8'))
        
        features = np.array(embeddings_data['features'])
        
        prediction, confidence = predict(model, scaler, features)
        
        inference_result = {
            "prediction": int(prediction),
            "confidence": float(confidence)
        }
        
        output_key = f"results/{session_id}/audio/{chunk_id}/inference.json"
        s3_output = f"s3://{input_bucket}/{output_key}"
        
        try:
            s3_client.put_object(
                Bucket=input_bucket,
                Key=output_key,
                Body=json.dumps(inference_result),
                ContentType='application/json'
            )
        except Exception as e:
            raise S3WriteError(f"Failed to write inference: {str(e)}")
        
        output_data = {
            "sessionId": session_id,
            "fileType": "audio",
            "chunkId": chunk_id,
            "s3Output": s3_output,
            "status": "success",
            "metadata": {
                "confidence": float(confidence),
                "prediction": int(prediction)
            },
            "error": None
        }
        print(json.dumps(output_data))
        sys.exit(0)
        
    except AudioInferenceError as e:
        output_data = {
            "sessionId": session_id,
            "fileType": "audio",
            "chunkId": chunk_id,
            "s3Output": "",
            "status": "failed",
            "metadata": {},
            "error": str(e)
        }
        print(json.dumps(output_data), file=sys.stderr)
        sys.exit(1)
    except S3WriteError as e:
        output_data = {
            "sessionId": session_id,
            "fileType": "audio",
            "chunkId": chunk_id,
            "s3Output": "",
            "status": "failed",
            "metadata": {},
            "error": str(e)
        }
        print(json.dumps(output_data), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        output_data = {
            "sessionId": session_id,
            "fileType": "audio",
            "chunkId": chunk_id,
            "s3Output": "",
            "status": "failed",
            "metadata": {},
            "error": f"AudioInferenceError: {str(e)}"
        }
        print(json.dumps(output_data), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
