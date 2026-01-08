# test_script.py
import json
import boto3
from video_splitter import lambda_handler  # Import your function

# Initialize S3 client
s3_client = boto3.client('s3')

# 1. Define a Mock Event (This matches the Contract Input)
test_event = {
  "sessionId": "test-session-001",
  "fileType": "video",
  "s3Input": "s3://deception-detection-bucket/dataset/video/truthful/TTTT_101_class_Truth_83.mkv", 
  "chunkSizeSeconds": 10,
  "metadata": {}
}

# 2. Define a Mock Context (Lambda expects this object, even if empty)
class MockContext:
    function_name = "test_function"
    memory_limit_in_mb = 128
    aws_request_id = "test-id-123"

# 3. Run the Handler
try:
    print("--- STARTING TEST ---")
    result = lambda_handler(test_event, MockContext())
    
    print("\n--- RESULT ---")
    print(json.dumps(result, indent=2))
    
    # 4. Display Video Chunks Information
    if result.get('status') == 'success' and result.get('videoChunks'):
        print("\n--- VIDEO CHUNKS CREATED ---")
        print(f"Total chunks: {len(result['videoChunks'])}")
        print(f"Has audio: {result['hasAudio']}")
        
        for idx, chunk_uri in enumerate(result['videoChunks'], 1):
            print(f"\nChunk {idx}: {chunk_uri}")
            
            # Parse S3 URI to get bucket and key
            bucket = chunk_uri.split('/')[2]
            key = '/'.join(chunk_uri.split('/')[3:])
            
            # Get file size
            try:
                response = s3_client.head_object(Bucket=bucket, Key=key)
                size_mb = response['ContentLength'] / (1024 * 1024)
                print(f"  - Size: {size_mb:.2f} MB")
                print(f"  - Content-Type: {response.get('ContentType', 'N/A')}")
            except Exception as e:
                print(f"  - Could not retrieve metadata: {e}")
        
        # 5. Option to download chunks locally
        print("\n--- DOWNLOAD OPTIONS ---")
        download = input("Download chunks to local directory? (y/n): ").lower()
        
        if download == 'y':
            import os
            download_dir = './downloaded_chunks'
            os.makedirs(download_dir, exist_ok=True)
            
            for idx, chunk_uri in enumerate(result['videoChunks'], 1):
                bucket = chunk_uri.split('/')[2]
                key = '/'.join(chunk_uri.split('/')[3:])
                local_filename = os.path.join(download_dir, os.path.basename(key))
                
                print(f"Downloading chunk {idx}...")
                s3_client.download_file(bucket, key, local_filename)
                print(f"  - Saved to: {local_filename}")
            
            print(f"\n✓ All chunks downloaded to: {download_dir}")
    
    elif result.get('status') == 'failed':
        print("\n--- PROCESSING FAILED ---")
        print(f"Error: {result.get('error')}")
    
except Exception as e:
    print("\n--- EXCEPTION OCCURRED ---")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()