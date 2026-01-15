import json
import torch
import os
from pathlib import Path

from inference import lambda_handler

# event structure 
event = {
    "sessionId": "test-session-002",
    "fileType": "tensor",
    "chunkId": "chunk_01",
    "s3InputTensor": "s3://deception-detection-bucket/test-session-001/video/chunk_34/feature.pt",
    "modelVersion": "video-model-v1.0",
    "metadata": {
        "originalResolution": {"width": 0, "height": 0},
        "numFrames": 64
        }
    }

response = lambda_handler(event, None)
print(response)
