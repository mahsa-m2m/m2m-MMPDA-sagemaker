import json
import torch
import os
from pathlib import Path

from preprocessing import lambda_handler

# event structure 
event = {
    "sessionId": "test-session-001",
    "chunkId": "chunk_34",
    "fileType": "video",
    "s3Input": "s3://deception-detection-bucket/dataset/video/deceptive/TTTT_1107_class_Deceptive_50.mkv", 
    "extractAudio": True,
    "resize": {"height": 224, "width": 224}
}

# Run handler
response = lambda_handler(event, None)
print(response)
