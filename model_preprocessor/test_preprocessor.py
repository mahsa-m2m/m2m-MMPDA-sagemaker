import json
import boto3
import logging
import sys
import os

from main import process_video_task 

logging.basicConfig(level=logging.INFO, force=True)

# Initialize S3 client
s3_client = boto3.client('s3')

test_event = {
  "sessionId": "test-session-01",
  "chunkId": "chunk_001",  
  "fileType": "video",
  "s3Input": "s3://deception-detection-bucket/dataset/video/deceptive/TTTT_10_class_Deceptive_54.mkv", 
  "extractAudio": True
}

# --- RUNNING THE TEST ---
try:
    print("\n=== STARTING LOCAL TEST ===")
    print(f"Input: {test_event['s3Input']}")
    
    # Call the logic function directly
    result = process_video_task(test_event)
    
    print("\n=== RAW RESULT ===")
    print(json.dumps(result, indent=2))
    
    if result.get('status') == 'success':
        output_uri = result.get('s3OutputVideo')
        print(f"\n=== VERIFYING OUTPUT: {output_uri} ===")
        
        if output_uri:
            # Parse S3 URI to get bucket and key
            bucket = output_uri.replace("s3://", "").split("/")[0]
            key = output_uri.replace(f"s3://{bucket}/", "")
            
            try:
                response = s3_client.head_object(Bucket=bucket, Key=key)
                size_bytes = response['ContentLength']
                size_mb = size_bytes / (1024 * 1024)
                
                print(f" SUCCESS: Output file found on S3.")
                print(f"   File: {key}")
                print(f"   Size: {size_mb:.2f} MB ({size_bytes} bytes)")
                print(f"   Type: .pt (Torch Tensor)")
            except Exception as e:
                print(f" ERROR: Result says success, but file not found on S3: {e}")
        else:
            print(" Result status is success, but 's3OutputVideo' is empty.")

    elif result.get('status') == 'failed':
        print("\n PROCESSING FAILED")
        print(f"Error Message: {result.get('error')}")
    
except Exception as e:
    print("\n EXCEPTION OCCURRED DURING TEST SCRIPT")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()