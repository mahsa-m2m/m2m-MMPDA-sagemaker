#!/usr/bin/env python3
import os
import json
import sys
import boto3
import filetype

class ModalityDetectionError(Exception):
    pass

class S3ReadError(Exception):
    pass

def parse_s3_path(s3_path: str) -> tuple:
    """Splits s3://bucket/key into bucket and key."""
    parts = s3_path.replace("s3://", "").split("/", 1)
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
            print(f"Detected MIME: {mime}")
            
            if mime.startswith('video'): return 'video'
            if mime.startswith('audio'): return 'audio'
            if mime == 'application/pdf': return 'text'

        # If 'filetype' returns None, it might be a plain text file (.txt, .csv)
        try:
            file_header_bytes.decode('utf-8')
            return 'text'
        except UnicodeDecodeError:
            pass

        return 'unknown'

def main():
    s3_client = boto3.client('s3')
    
    # Environment Variables
    session_id = os.environ.get("SESSION_ID", "unknown")
    s3_input = os.environ.get("S3_INPUT", "")
    
    try:
        # Input Validation
        if not s3_input:
            raise ModalityDetectionError("Missing S3_INPUT")
        
        input_bucket, input_key = parse_s3_path(s3_input)
        
        # We don't download the whole file to temp, only requires the first 2KB. 
        try:
            response = s3_client.get_object(
                Bucket=input_bucket, 
                Key=input_key, 
                Range='bytes=0-2047'
            )
            file_header = response['Body'].read()
        except Exception as e:
            raise S3ReadError(f"Failed to read file header from S3: {str(e)}")
            
        # Execute Domain Logic
        try:
            detector = ModalityDetector()
            modality = detector.detect(file_header)
            
            if modality == 'unknown': ## FATAL ERROR - INPUT IS NOT VALID
                pass
                
        except Exception as e:
            raise ModalityDetectionError(f"Failed to analyze file bytes: {str(e)}")
        
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
        
        # stdout
        print(json.dumps(output_data))
        sys.exit(0)
        
    # Error Handling
    except ModalityDetectionError as e:
        output_data = {
            "sessionId": session_id,
            "s3Input": s3_input,
            "detectedModality": "unknown",
            "status": "failed",
            "metadata": {},
            "error": str(e)
        }
        print(json.dumps(output_data), file=sys.stderr)
        sys.exit(1)

    except S3ReadError as e:
        output_data = {
            "sessionId": session_id,
            "s3Input": s3_input,
            "detectedModality": "unknown",
            "status": "failed",
            "metadata": {},
            "error": str(e)
        }
        print(json.dumps(output_data), file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        output_data = {
            "sessionId": session_id,
            "s3Input": s3_input,
            "detectedModality": "unknown",
            "status": "failed",
            "metadata": {},
            "error": f"UnhandledException: {str(e)}"
        }
        print(json.dumps(output_data), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
