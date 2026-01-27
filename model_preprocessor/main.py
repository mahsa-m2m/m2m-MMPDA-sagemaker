#!/usr/bin/env python
import sys
import os
import json
import boto3
import logging
import tempfile
import torch
from typing import Dict, Any

# Immediate output for CloudWatch
sys.stdout.write("=== CONTAINER SCRIPT STARTED ===\n")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

try:
    from utils import parse_s3_path, InvalidInputError, S3WriteError, VideoPreprocessError
    from VideoFeatureExtractor import VideoFeatureExtractor
    import config 
except ImportError as e:
    logger.error(f"Failed to import modules: {str(e)}")
    sys.exit(1)

preprocessing_engine = None

def get_preprocessing_engine():
    global preprocessing_engine
    if preprocessing_engine is None:
        logger.info("Initializing VideoFeatureExtractor...")
        try:
            face_landmark_path = os.environ.get("FACE_LANDMARK_PATH", "s3://coyote-deception-detection-platform/models/video/face_landmarker.task")
            if face_landmark_path.startswith("s3://"):
                logger.info(f"Found S3 face model path: {face_landmark_path}")
                try:
                    s3_client = boto3.client('s3')
                    bucket, key = parse_s3_path(face_landmark_path)
                    
                    local_model_path = "/tmp/face_landmark_model.pth"
                    
                    logger.info(f"Downloading model from bucket: {bucket}, key: {key}")
                    s3_client.download_file(bucket, key, local_model_path)
                    
                    config.FACE_LANDMARK_PATH = local_model_path
                    logger.info(f"Face Model successfully downloaded to {local_model_path}")
                    
                except Exception as e:
                    logger.critical(f"Failed to download face model from S3: {e}")
                    raise e
            preprocessing_engine = VideoFeatureExtractor()
        except Exception as e:
            logger.error(f"Failed to initialize engine: {str(e)}")
            raise
    return preprocessing_engine

def process_video_task(event: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Starting video processing task")
    
    s3_client = boto3.client('s3')
    
    # Validate Inputs
    session_id = event.get('sessionId')
    chunk_id = event.get('chunkId')
    s3_input = event.get('s3Input')
    extract_audio = event.get('extractAudio', False)
    
    if not all([session_id, chunk_id, s3_input]):
        raise InvalidInputError("Missing required fields (sessionId, chunkId, or s3Input)")

    # Download from S3
    logger.info("Parsing S3 path")
    input_bucket, input_key = parse_s3_path(s3_input)
    _, file_extension = os.path.splitext(input_key)
    if not file_extension: file_extension = '.mp4'

    logger.info(f"Downloading file from S3: {input_bucket}/{input_key}")
    
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
        try:
            s3_client.download_file(input_bucket, input_key, tmp_file.name)
            input_path = tmp_file.name
            logger.info(f"Downloaded video to {input_path}")
        except Exception as e:
            raise InvalidInputError(f"Failed to download file: {str(e)}")

    # Process Video
    try:
        engine = get_preprocessing_engine()
        features = engine.process_video(input_path, extract_audio)
        logger.info("Feature extraction successful")
    except Exception as e:
        if os.path.exists(input_path): os.unlink(input_path)
        raise 

    # Save and Upload Results
    s3_key_feature = f"{session_id}/video/{chunk_id}/feature.pt"
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_out_feat:
            torch.save(features, tmp_out_feat.name, pickle_protocol=4)
            output_feature_path = tmp_out_feat.name
        
        logger.info(f"Uploading results to s3://{input_bucket}/{s3_key_feature}")
        s3_client.upload_file(output_feature_path, input_bucket, s3_key_feature)
        
        # Cleanup output file
        os.unlink(output_feature_path)
        
    except Exception as e:
        raise S3WriteError(f"Failed to save/upload results: {str(e)}")
    finally:
        # Cleanup input file
        if os.path.exists(input_path): os.unlink(input_path)

    # Return Success
    return {
        "sessionId": session_id,
        "chunkId": chunk_id,
        "s3OutputVideo": f"s3://{input_bucket}/{s3_key_feature}",
        "status": "success",
        "metadata": {
            "numFrames": config.NUM_FRAMES
        }
    }

def main():
    sys.stdout.write("=== MAIN FUNCTION CALLED ===\n")
    logger.info("Video Preprocessing Container Starting")
    
    try:
        logger.info("Building event from environment variables")
        event = {
            "sessionId": os.environ.get("SESSION_ID"),
            "chunkId": os.environ.get("CHUNK_ID", "chunk_00"), # Default if not provided
            "s3Input": os.environ.get("S3_INPUT"),
            "fileType": "video",
            "extractAudio": os.environ.get("EXTRACT_AUDIO", "true").lower() == "true"
        }
        
        result = process_video_task(event)
        
        logger.info("Task completed successfully")
        logger.info(f"Result: {json.dumps(result)}")
        sys.exit(0)

    except (InvalidInputError, VideoPreprocessError, S3WriteError) as e:
        logger.error(f"Known error occurred: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()