import json
import torch
import os
from pathlib import Path

from preprocessing import lambda_handler

# Mock event structure 
event = {
    "sessionId": "test-session-001",
    "chunkId": "chunk_33",
    "fileType": "video",
    "s3Input": "s3://deception-detection-bucket/dataset/video/truthful/TTTT_1006_class_Truth_33.mkv", 
    "extractAudio": True,
    "resize": {"height": 224, "width": 224}
}

# Run handler
response = lambda_handler(event, None)
print(response)
