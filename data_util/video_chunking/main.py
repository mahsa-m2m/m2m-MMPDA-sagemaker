import sys
import os

# Immediate output for CloudWatch
sys.stdout.write("=== VIDEO CHUNKING CONTAINER STARTED ===\n")

import json
import boto3
import logging
import shutil
from pathlib import Path
from typing import Dict, Any


# Configure Logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.info("Logging system initialized")

# Import logic modules
try:
    from utils import parse_s3_path, VideoChunkingError, S3WriteError, InvalidInputError
    from VideoChunker import VideoChunker
    logger.info("Successfully imported utility and logic modules")
except ImportError as e:
    logger.error(f"Failed to import modules: {str(e)}")
    sys.exit(1)

chunker = None

def get_video_chunker():
    global chunker
    if chunker is None:
        logger.info("Initializing VideoChunker")
        try:
            chunker = VideoChunker()
            logger.info("VideoChunker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VideoChunker: {str(e)}")
            import traceback
            logger.error(f"Initialization traceback: {traceback.format_exc()}")
            raise
    else:
        logger.debug("Reusing existing VideoChunker instance")
    return chunker

def video_chunking(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main orchestration logic:
    1. Download from S3
    2. Call VideoChunker to split
    3. Upload chunks to S3
    """
    logger.info("Starting video processing task")
    logger.debug(f"Input event: {json.dumps(event, indent=2)}")

    # Initialize S3
    s3_client = boto3.client('s3')
    
    # Extract details from event
    session_id = event.get('sessionId')
    if not session_id:
        raise InvalidInputError("Missing sessionId")

    s3_input = event.get('s3Input')
    if not s3_input:
        raise InvalidInputError("Missing s3Input")

    chunk_size = int(event.get('chunkSizeSeconds', 10))

    # Parse S3 Path
    input_bucket, input_key = parse_s3_path(s3_input)
    logger.info(f"Target Bucket: {input_bucket}, Key: {input_key}")

    # Prepare local paths
    local_filename = os.path.basename(input_key)
    local_input_path = os.path.join('/tmp', local_filename)
    output_dir = os.path.join('/tmp', 'chunks')

    # Clean /tmp directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Download ---
    logger.info(f"Downloading file from S3: {s3_input}")
    try:
        s3_client.download_file(input_bucket, input_key, local_input_path)
        file_size = os.path.getsize(local_input_path)
        logger.info(f"Download complete. Size: {file_size / (1024*1024):.2f} MB")
    except Exception as e:
        raise InvalidInputError(f"Failed to download from S3: {str(e)}")

    # --- Processing (Class VideoChunker()) ---
    logger.info("Initializing VideoChunker")
    video_chunker = get_video_chunker()
    
    logger.info(f"Splitting video with chunk size: {chunk_size}s")
    try:
        # FFmpeg work
        local_chunk_files = video_chunker.split_video_precise(local_input_path, output_dir, chunk_size)
        has_audio = video_chunker.check_has_audio(local_input_path)
        logger.info(f"Splitting complete. Generated {len(local_chunk_files)} chunks.")
    except Exception as e:
        logger.error(f"Video chunking failed: {str(e)}")
        raise VideoChunkingError(str(e))

    # --- Upload ---
    uploaded_uris = []
    logger.info("Uploading chunks to S3")
    try:
        for local_file in local_chunk_files:
            fname = os.path.basename(local_file)
            # Output structure: results/{session_id}/video/{filename}
            s3_output_key = f"results/{session_id}/video/{fname}"
            
            s3_client.upload_file(local_file, input_bucket, s3_output_key)
            
            s3_uri = f"s3://{input_bucket}/{s3_output_key}"
            uploaded_uris.append(s3_uri)
            logger.debug(f"Uploaded: {fname}")

    except Exception as e:
        raise S3WriteError(f"Failed to upload chunks: {str(e)}")

    # Prepare Success Result
    result = {
        "sessionId": session_id,
        "fileType": "video",
        "videoChunks": uploaded_uris,
        "hasAudio": has_audio,
        "status": "success",
        "metadata": {
            "chunkCount": len(uploaded_uris),
            "chunkSizeSeconds": chunk_size
        }
    }
    
    logger.info("Video processing completed successfully")
    return result

def main():
    """
    Reads Env Vars -> Builds Event -> Calls Processor -> Exits
    """
    sys.stdout.write("=== MAIN FUNCTION CALLED ===\n")
    logger.info("Video Processing Container Starting")

    # Log environment variables
    logger.info("Environment variables:")
    logger.info(f"  SESSION_ID: {os.environ.get('SESSION_ID', 'NOT SET')}")
    logger.info(f"  S3_INPUT: {os.environ.get('S3_INPUT', 'NOT SET')}")
    logger.info(f"  CHUNK_SIZE_SECOND: {os.environ.get('CHUNK_SIZE', 'NOT SET')}")
    logger.info(f"  AWS_DEFAULT_REGION: {os.environ.get('AWS_DEFAULT_REGION', 'NOT SET')}")    
    event = {}

    try:
        logger.info("Building event from environment variables")
        event = {
            "sessionId": os.environ.get("SESSION_ID"),
            "fileType": "video",
            "s3Input": os.environ.get("S3_INPUT"),
            "chunkSizeSeconds": os.environ.get("CHUNK_SIZE", 10)
        }
        logger.info(f"Event built: sessionId={event.get('sessionId')}, s3Input={event.get('s3Input')}")
        
        logger.info("Calling video_chunking()")
        result = video_chunking(event)
        
        logger.info("Task completed successfully")
        logger.info(f"Result: {json.dumps(result)}")
        sys.exit(0)
        
    except InvalidInputError as e:
        logger.error("InvalidInputError occurred")
        logger.error(f"  Session ID: {event.get('sessionId', 'unknown')}")
        logger.error(f"  S3 Input: {event.get('s3Input', 'unknown')}")
        logger.error(f"  Error: {str(e)}")
        sys.exit(1)
    except S3WriteError as e:
        logger.error("S3WriteError occurred")
        logger.error(f"  Session ID: {event.get('sessionId', 'unknown')}")
        logger.error(f"  S3 Input: {event.get('s3Input', 'unknown')}")
        logger.error(f"  Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected Exception occurred")
        logger.error(f"  Exception type: {type(e).__name__}")
        logger.error(f"  Session ID: {event.get('sessionId', 'unknown')}")
        logger.error(f"  S3 Input: {event.get('s3Input', 'unknown')}")
        logger.error(f"  Error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        logger.info("=== ENTRY POINT REACHED ===")
        main()
    except Exception as e:
        logger.error(f"Fatal error in entry point: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


