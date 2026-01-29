import os
import sys
from unittest.mock import MagicMock, patch

os.environ["TABLE_NAME"] = "TestTable"
os.environ["REPORT_BUCKET"] = "TestBucket"

mock_s3 = MagicMock()
mock_ddb = MagicMock()

def print_report_content(**kwargs):
    print("\n--- [MOCK S3] Uploading File ---")
    print(f"Bucket: {kwargs.get('Bucket')}")
    print(f"Key:    {kwargs.get('Key')}")
    print("--- CONTENT START ---")
    print(kwargs.get('Body'))
    print("--- CONTENT END ---\n")
    return {}

mock_s3.put_object.side_effect = print_report_content

with patch('boto3.client') as mock_boto:
    def side_effect(service_name):
        if service_name == 's3': return mock_s3
        if service_name == 'dynamodb': return mock_ddb
        return MagicMock()
    
    mock_boto.side_effect = side_effect
    
    # from generator_VA import lambda_handler
    from generator_T import lambda_handler

    test_event = {
      "session_id": "session_123",
      "file_name": "interview.mp4",
      "chunk_results": [
        {"chunkId": "1", "fileType": "audio", "metadata": {"prediction": "0", "confidence": 0.9, "durationSeconds": 10}},
        {"chunkId": "2", "fileType": "video", "metadata": {"prediction": "1", "confidence": 0.8, "durationSeconds": 10}},
        {"chunkId": "3", "fileType": "video", "metadata": {"prediction": "1", "confidence": 0.5, "durationSeconds": 5}}
      ]
    }

    print(">>> RUNNING LAMBDA HANDLER LOCALLY")
    response = lambda_handler(test_event, None)
    print(f">>> RESPONSE: {response}")