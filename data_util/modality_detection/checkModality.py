#!/usr/bin/env python3
import os
import json
import sys
import boto3
import filetype
import logging
from botocore.exceptions import ClientError 

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

class ModalityDetectionError(Exception):
    pass

class S3ReadError(Exception):
    pass

def parse_s3_path(s3_path: str) -> tuple:
    """Splits s3://bucket/key into bucket and key."""
    if not s3_path.startswith("s3://"):
        raise ValueError("Invalid S3 URI scheme")
        
    parts = s3_path.replace("s3://", "").split("/", 1)
    if len(parts) < 2:
        raise ValueError("Invalid S3 path format")
        
    return parts[0], parts[1]

class ModalityDetector:
    def __init__(self):
        pass
    
    def detect(self, file_header_bytes: bytes) -> str:
        """
        Determines if content is Video, Audio, or Text using pure python methods.
        """
        # Guess the file type using binary signatures
        kind = filetype.guess(file_header_bytes)

        if kind:
            mime = kind.mime
            logger.info(f"Detected MIME: {mime}")
            
            if mime.startswith('video'): return 'video'
            if mime.startswith('audio'): return 'audio'
            if mime == 'application/pdf': return 'text'

        # If 'filetype' returns None, it might be a plain text file (.txt, .csv)
        try:
            # file_header_bytes.decode('utf-8')
            file_header_bytes.decode('utf-8', errors='ignore')
            return 'text'
        except UnicodeDecodeError:
            pass

        return 'unknown'

def lambda_handler(event, context):
    try:
        logger.info(f"Received event - Check Modality: {json.dumps(event)}")
        
        session_id = event.get("sessionId", os.environ.get("SESSION_ID", "unknown"))
        s3_input = event.get("s3Input", os.environ.get("S3_INPUT", ""))
        
        # Input Validation
        if not s3_input:
            raise ModalityDetectionError("Missing s3Input in event or environment")
        
        input_bucket, input_key = parse_s3_path(s3_input)
        
        try:
            response = s3_client.get_object(
                Bucket=input_bucket, 
                Key=input_key, 
                Range='bytes=0-2047'
            )
            file_header = response['Body'].read()
        except ClientError as e:
            logger.error(f"AWS S3 Error: {e}")
            raise S3ReadError(f"Failed to read file header from S3: {str(e)}")
        except Exception as e:
            raise S3ReadError(f"General S3 read error: {str(e)}")
            
        detector = ModalityDetector()
        modality = detector.detect(file_header)
        
        if modality == 'unknown': 
            logger.warning(f"Modality detection resulted in 'unknown' for {s3_input}")
        
        # Construct Success Output
        output_data = {
            "sessionId": session_id,
            "s3Input": s3_input,
            "detectedModality": modality, 
            "status": "success",
            "metadata": {
                "mimeType": filetype.guess(file_header).mime if filetype.guess(file_header) else "text/plain"
            },
            "error": None
        }
        
        return {
            "statusCode": 200,
            "body": json.dumps(output_data)
        }

    except (ModalityDetectionError, S3ReadError, ValueError) as e:
        logger.error(f"Known Error: {str(e)}")
        error_output = {
            "sessionId": event.get("sessionId", "unknown"),
            "status": "failed",
            "error": str(e)
        }
        return {
            "statusCode": 400, # Bad Request
            "body": json.dumps(error_output)
        }

    except Exception as e:
        logger.error(f"Unhandled Lambda Exception: {str(e)}", exc_info=True)
        error_output = {
            "sessionId": event.get("sessionId", "unknown"),
            "status": "failed",
            "error": "Internal Server Error"
        }
        return {
            "statusCode": 500, # Internal Error
            "body": json.dumps(error_output)
        }



test_event = {
    "sessionId": "test-123",
    "s3Input": "s3://coyote-deception-detection-platform/uploads/session_b107aaefa0a2_1770826888/text/text_deceptive.txt"
}

class MockContext:
    function_name = "test_modality_detector"

if __name__ == "__main__":
    response = lambda_handler(test_event, MockContext())
    print(json.dumps(response, indent=2))
