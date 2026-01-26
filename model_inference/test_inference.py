import json
import boto3
import logging
import sys
import os

from main import process_inference_task 

logging.basicConfig(level=logging.INFO, force=True)

s3_client = boto3.client('s3')

test_event = {
  "sessionId": "test-session-inference-01",
  "chunkId": "chunk_001",  
  "fileType": "video",
  "s3InputTensor": "s3://deception-detection-bucket/test-session-01/video/chunk_001/feature.pt", 
  "metadata": {}
}

# --- RUNNING THE TEST ---
try:
    print("\n=== STARTING LOCAL INFERENCE TEST ===")
    print(f"Input Tensor: {test_event['s3InputTensor']}")
    
    result = process_inference_task(test_event)
    
    print("\n=== RAW RESULT ===")
    print(json.dumps(result, indent=2))
    
    # --- VERIFYING S3 OUTPUT ---
    if result.get('status') == 'success':
        output_uri = result.get('s3Output')
        print(f"\n=== VERIFYING OUTPUT: {output_uri} ===")
        
        if output_uri:
            bucket = output_uri.replace("s3://", "").split("/")[0]
            key = output_uri.replace(f"s3://{bucket}/", "")
            
            try:
                # Check if file exists
                response = s3_client.head_object(Bucket=bucket, Key=key)
                size_bytes = response['ContentLength']
                
                print(f" SUCCESS: Output file found on S3.")
                print(f"   File: {key}")
                print(f"   Size: {size_bytes} bytes")
                
                # Download and read the JSON content to show the prediction
                print("   Downloading content...")
                file_obj = s3_client.get_object(Bucket=bucket, Key=key)
                file_content = file_obj['Body'].read().decode('utf-8')
                json_content = json.loads(file_content)
                
                print("\n   --- INFERENCE CONTENT ---")
                print(f"   Prediction: {json_content['metadata']['prediction']} (0=Truthful, 1=Deceptive)")
                print(f"   Confidence: {json_content['metadata']['confidence']:.4f}")
                
            except Exception as e:
                print(f" ERROR: Result says success, but could not read file on S3: {e}")
        else:
            print(" Result status is success, but 's3Output' is empty.")

    elif result.get('status') == 'failed':
        print("\n PROCESSING FAILED")
        print(f"Error Message: {result.get('error')}")
    
except Exception as e:
    print("\n EXCEPTION OCCURRED DURING TEST SCRIPT")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()