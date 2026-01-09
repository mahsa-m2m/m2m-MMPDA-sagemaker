import json
import torch
import os
from pathlib import Path

from preprocessing import lambda_handler

# Mock event structure based on your code
event = {
    "sessionId": "test-session-001",
    "chunkId": "chunk_01",
    "fileType": "video",
    "s3Input": "s3://deception-detection-bucket/dataset/video/truthful/TTTT_1002_class_Truth_33.mkv", # Make sure this file exists in S3
    "extractAudio": True,
    "resize": {"height": 224, "width": 224}
}

# Run handler
response = lambda_handler(event, None)
print(response)
