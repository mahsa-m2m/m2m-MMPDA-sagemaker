import json
import boto3
import logging
import sys

from main import video_chunking 

logging.basicConfig(level=logging.INFO, force=True)

# Initialize S3 client
s3_client = boto3.client('s3')

# Define a Mock Event
test_event = {
  "sessionId": "test-session-01",
  "fileType": "video",
  "s3Input": "s3://deception-detection-bucket/dataset/video/truthful/TTTT_104_class_Truth_47.mkv", 
  "chunkSizeSeconds": 10,
  "metadata": {}
}

# Run the Handler
try:
    print("\n--- STARTING TEST ---")
    
    result = video_chunking(test_event)
    
    print("\n--- RESULT ---")
    print(json.dumps(result, indent=2))
    
    # Display Video Chunks Information
    if result.get('status') == 'success' and result.get('videoChunks'):
        print("\n--- VIDEO CHUNKS CREATED ---")
        print(f"Total chunks: {len(result['videoChunks'])}")
        print(f"Has audio: {result['hasAudio']}")
        
        for idx, chunk_uri in enumerate(result['videoChunks'], 1):
            print(f"\nChunk {idx}: {chunk_uri}")
            
            # Parse S3 URI to get bucket and key
            bucket = chunk_uri.replace("s3://", "").split("/")[0]
            key = chunk_uri.replace(f"s3://{bucket}/", "")
            
            # Get file size
            try:
                response = s3_client.head_object(Bucket=bucket, Key=key)
                size_mb = response['ContentLength'] / (1024 * 1024)
                print(f"  - Size: {size_mb:.2f} MB")
            except Exception as e:
                print(f"  - Could not retrieve metadata: {e}")

    elif result.get('status') == 'failed':
        print("\n--- PROCESSING FAILED ---")
        print(f"Error: {result.get('error')}")
    
except Exception as e:
    print("\n--- EXCEPTION OCCURRED ---")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()