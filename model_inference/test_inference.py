import json
import torch
import os
from pathlib import Path

from inference import lambda_handler

# event structure 
event = {
    "sessionId": "test-session-001",
    "fileType": "tensor",
    "chunkId": "chunk_01",
    "s3InputTensor": "s3://deception-detection-bucket/dataset/video/precomputed_features/test/TTTT_1016_class_Truth_19.pt",
    "modelVersion": "video-model-v1.0",
    "metadata": {
        "originalResolution": {"width": 1920, "height": 1080},
        "numFrames": 120
        }
    }

response = lambda_handler(event, None)
print(response)
