#!/usr/bin/env python
# from . import config
# from .fusionEngine import SoftVotingFusion
import fusion_config as config
from fusionEngine import SoftVotingFusion
import sys
import os
import json
import boto3
import logging
from typing import Dict, Any

sys.stdout.write("=== FUSION CONTAINER SCRIPT STARTED ===\n")


try:
    from utils import parse_s3_path, InvalidInputError, S3WriteError
except ImportError:
    logger.error("Failed to import utility functions. Ensure 'utils.py' is in the same directory or PYTHONPATH.")
    sys.exit(1)

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

# To confirm logging works
logger.info("Logging system initialized")

# Initialize the engine once 
fusion_engine = None
# engine = SoftVotingFusion(weights=config.FUSION_WEIGHTS)

### AWS
def get_fusion_engine():
    global fusion_engine
    if fusion_engine is None:
        logger.info(f"Initializing SoftVotingFusion with weights: {config.FUSION_WEIGHTS}")
        try:
            fusion_engine = SoftVotingFusion(weights=config.FUSION_WEIGHTS)
            logger.info("SoftVotingFusion initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SoftVotingFusion: {str(e)}")
            import traceback
            logger.error(f"Initialization traceback: {traceback.format_exc()}")
            raise
    else:
        logger.debug("Reusing existing SoftVotingFusion instance")
    return fusion_engine

def load_prediction_from_s3(s3_client, s3_path: str) -> Dict[str, Any]:
    """Load model prediction results from S3"""
    logger.info(f"Loading prediction from: {s3_path}")
    bucket, key = parse_s3_path(s3_path)
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Successfully loaded prediction from {s3_path}")
        logger.debug(f"Prediction data: {json.dumps(data, indent=2)}")
        return data
    except Exception as e:
        logger.error(f"Failed to load prediction from {s3_path}: {str(e)}")
        raise InvalidInputError(f"Failed to load prediction: {str(e)}")


### Fusion
def _extract_probs_from_response(response_data):
    """
    Parses the specific model output JSON to create a probability distribution.
    Class "0" = Truthful, Class "1" = Deceptive.
    Returns: [Prob_Truthful, Prob_Deceptive]
    """
    logger.debug(f"Extracting probabilities from response: {response_data}")

    if not response_data or 'metadata' not in response_data:
        logger.warning("Response data missing or no metadata field")
        return None

    meta = response_data['metadata']
    prediction = str(meta.get('prediction', '0')) # "0" or "1"
    confidence = float(meta.get('confidence', 0.5))
    
    # Normalize into vector: [Prob_Truthful, Prob_Deceptive]
    if prediction == '1':
        # Deceptive prediction: confidence is for deceptive class
        probs = [1.0 - confidence, confidence]
    else:
        # Truthful prediction: confidence is for truthful class
        probs = [confidence, 1.0 - confidence]
    
    logger.debug(f"Probability vector: {probs}")
    return probs

def determine_primary_modality(video_probs, audio_probs, winning_index, weights):
    """
    Calculates which modality contributed more to the winning label.
    """
    # If Audio is missing, Video is automatically the primary
    if audio_probs is None:
        logger.debug("Audio missing, video is primary modality")
        return "video"

    # Get the probability each modality assigned to the WINNING class
    v_contribution = video_probs[winning_index] * weights.get('video', 0.0)
    a_contribution = audio_probs[winning_index] * weights.get('audio', 0.0)

    logger.debug(f"Video contribution: {v_contribution}, Audio contribution: {a_contribution}")

    # Compare weighted contributions
    if v_contribution >= a_contribution:
        return "video"
    else:
        return "audio"

def process_fusion_request(video_response, audio_response=None):
    logger.info("Starting fusion processing task")
    try:
        # Extract Inputs
        v_probs = _extract_probs_from_response(video_response)
        a_probs = _extract_probs_from_response(audio_response)

        # Validation
        if audio_response and (video_response.get('chunkId') != audio_response.get('chunkId')):
            return {"status": "error", "message": "Chunk ID mismatch"}

        engine = get_fusion_engine()
        # Run Inference
        result = engine.predict(video_probs=v_probs, audio_probs=a_probs)

        if result["status"] == "error":
            return result

        # Determine Primary Modality 

        primary = determine_primary_modality(
            video_probs=v_probs, 
            audio_probs=a_probs, 
            winning_index=result['label_index'], 
            weights=config.FUSION_WEIGHTS
        )

        # Format Final Output
        return {
            "primaryModality": primary, # "video" or "audio",
            "prediction": str(result["label_index"]),
            "confidence": result["confidence"]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    """
    Loads from S3, calls process_fusion_request, writes to S3
    """
    sys.stdout.write("=== MAIN FUNCTION CALLED ===\n")
    logger.info("Fusion Module Container Starting")
    
    # Log environment variables
    logger.info("Environment variables:")
    logger.info(f"  SESSION_ID: {os.environ.get('SESSION_ID', 'NOT SET')}")
    logger.info(f"  CHUNK_ID: {os.environ.get('CHUNK_ID', 'NOT SET')}")
    logger.info(f"  VIDEO_PREDICTION: {os.environ.get('VIDEO_PREDICTION', 'NOT SET')}")
    logger.info(f"  AUDIO_PREDICTION: {os.environ.get('AUDIO_PREDICTION', 'NOT SET')}")
    
    s3_client = boto3.client('s3')
    
    try:
        # Get environment variables
        session_id = os.environ.get("SESSION_ID")
        chunk_id = os.environ.get("CHUNK_ID")
        video_pred_path = os.environ.get("VIDEO_PREDICTION")
        audio_pred_path = os.environ.get("AUDIO_PREDICTION")
        
        if not session_id or not chunk_id or not video_pred_path:
            logger.error("Missing required environment variables")
            sys.exit(1)
        
        logger.info(f"Processing fusion for session={session_id}, chunk={chunk_id}")
        
        # Load video prediction from S3
        logger.info(f"Loading video prediction from: {video_pred_path}")
        video_bucket, video_key = parse_s3_path(video_pred_path)
        video_obj = s3_client.get_object(Bucket=video_bucket, Key=video_key)
        video_response = json.loads(video_obj['Body'].read().decode('utf-8'))
        logger.info("Video prediction loaded successfully")
        
        # Load audio prediction from S3
        audio_response = None
        if audio_pred_path:
            logger.info(f"Loading audio prediction from: {audio_pred_path}")
            audio_bucket, audio_key = parse_s3_path(audio_pred_path)
            audio_obj = s3_client.get_object(Bucket=audio_bucket, Key=audio_key)
            audio_response = json.loads(audio_obj['Body'].read().decode('utf-8'))
            logger.info("Audio prediction loaded successfully")
        else:
            logger.info("No audio prediction provided")
        
        # Call fusion function
        logger.info("Calling process_fusion_request")
        result = process_fusion_request(video_response, audio_response)
        logger.info(f"Fusion result: {result}")
        
        # Check for errors
        if result.get("status") == "error":
            logger.error(f"Fusion failed: {result.get('message')}")
            sys.exit(1)
        
        # Prepare output for S3
        output_data = {
            "sessionId": session_id,
            "chunkId": chunk_id,
            "primaryModality": result["primaryModality"],
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        }
        
        # Write to S3
        output_key = f"results/{session_id}/fusion/{chunk_id}/prediction.json"
        output_path = f"s3://{video_bucket}/{output_key}"
        
        logger.info(f"Writing fusion result to: {output_path}")
        s3_client.put_object(
            Bucket=video_bucket,
            Key=output_key,
            Body=json.dumps(output_data, indent=2),
            ContentType='application/json'
        )
        logger.info("Fusion result written successfully")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
