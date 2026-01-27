#!/usr/bin/env python
import sys
import os
import json
import boto3
import logging
import tempfile
from typing import Dict, Any

try:
    from utils import parse_s3_path, S3WriteError, InvalidInputError
    from VideoInference import FusionInference
    import config
except ImportError as e:
    sys.stderr.write(f"❌ Import Error: {e}\n")
    sys.exit(1)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

inference_engine = None

def get_inference_engine():
    global inference_engine
    if inference_engine is None:
        
        model_path_s3 = os.environ.get("MODEL_WEIGHTS_PATH", "s3://coyote-deception-detection-platform/models/video/2025-12-19-15-46-02-390.pt")
        if model_path_s3.startswith("s3://"):
            logger.info(f"Found S3 model path: {model_path_s3}")
            try:
                s3_client = boto3.client('s3')
                bucket, key = parse_s3_path(model_path_s3)
                
                local_model_path = "/tmp/video_model.pth"
                
                logger.info(f"Downloading model from bucket: {bucket}, key: {key}")
                s3_client.download_file(bucket, key, local_model_path)
                
                config.MODEL_WEIGHTS_PATH = local_model_path
                logger.info(f"Model successfully downloaded to {local_model_path}")
                
            except Exception as e:
                logger.critical(f"Failed to download model from S3: {e}")
                raise e
        
        resnet_path_s3 = os.environ.get("RESNET_LSTM_PATH", "s3://coyote-deception-detection-platform/models/video/resnet18-f37072fd.pth")
        if resnet_path_s3.startswith("s3://"):
            logger.info(f"Found S3 ResNet path: {resnet_path_s3}")
            try:
                bucket_res, key_res = parse_s3_path(resnet_path_s3)
                local_resnet_path = "/tmp/resnet18_lstm.pth"
                
                # Check if already exists
                if not os.path.exists(local_resnet_path):
                    logger.info(f"Downloading ResNet from bucket: {bucket_res}, key: {key_res}")
                    s3_client.download_file(bucket_res, key_res, local_resnet_path)
                
                config.RESNET18_LSTM_PATH = local_resnet_path
                logger.info(f"ResNet successfully configured at {local_resnet_path}")

            except Exception as e:
                logger.critical(f"Failed to download ResNet from S3: {e}")
                raise e
        
        logger.info("Initializing FusionInference Engine...")
        inference_engine = FusionInference()
    return inference_engine

    #     logger.info("Initializing FusionInference Engine...")
    #     inference_engine = FusionInference()
    # return inference_engine

def process_inference_task(event: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Starting inference task")
    
    s3_client = boto3.client('s3')

    # Prepare Response Template
    response_template = {
        "sessionId": event.get('sessionId', 'unknown'),
        "fileType": event.get('fileType', 'unknown'),
        "chunkId": event.get('chunkId', 'unknown'),
        "s3Output": "",
        "status": "failed",
        "metadata": event.get('metadata', {}),
        "error": None
    }

    chunk_path = None

    try:
        # Validate Input
        required_keys = ['sessionId', 'chunkId', 's3InputTensor']
        for key in required_keys:
            if key not in event:
                raise KeyError(key)

        input_tensor_s3_url = event['s3InputTensor']
        session_id = event['sessionId']
        chunk_id = event['chunkId']
        
        logger.info(f"Downloading input tensor from: {input_tensor_s3_url}")
        input_bucket, input_key = parse_s3_path(input_tensor_s3_url)

        # Download .pt file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            s3_client.download_file(input_bucket, input_key, tmp_file.name)
            chunk_path = tmp_file.name
        
        # Run Inference
        model = get_inference_engine()
        
        batch_results = model.predict_batch_silent(
            [chunk_path], 
            batch_size=config.BATCH_SIZE
        )

        if not batch_results:
            raise ValueError("Model inference returned no results.")

        # Format Results
        prediction_data = batch_results[0]
        response_template['status'] = 'success' 
        output_key = f"results/{session_id}/video/{chunk_id}/inference.json"
        response_template['s3Output'] = f"s3://{input_bucket}/{output_key}"
        
        response_template['metadata']['prediction'] = prediction_data['predicted_label']
        
        if prediction_data['predicted_label'] == "0":
            response_template['metadata']['confidence'] = prediction_data['truthful_prob']
        else:
            response_template['metadata']['confidence'] = prediction_data['deceptive_prob']

        # Upload Result to S3
        logger.info(f"Uploading results to s3://{input_bucket}/{output_key}")
        try:
            s3_client.put_object(
                Bucket=input_bucket,
                Key=output_key,
                Body=json.dumps(response_template),
                ContentType='application/json'
            )  
        except Exception as e:
            raise S3WriteError(f"Failed to write results: {str(e)}")

        return response_template  
    
    except KeyError as e:
        logger.error(f"Missing key: {e}")
        response_template['error'] = f"Missing required field: {str(e)}"
        return response_template

    except S3WriteError as e:
        logger.error(f"S3 Error: {e}")
        response_template['error'] = str(e)
        return response_template

    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        response_template['error'] = f"InferenceError: {str(e)}"
        return response_template

    finally:
        # Cleanup temp file
        if chunk_path and os.path.exists(chunk_path):
            os.unlink(chunk_path)

def main():
    sys.stdout.write("=== MAIN FUNCTION CALLED ===\n")
    logger.info("Video Inference Container Starting")
    
    # Construct Event from Environment Variables
    try:
        logger.info("Building event from environment variables")
        event = {
            "sessionId": os.environ.get("SESSION_ID"),
            "chunkId": os.environ.get("CHUNK_ID", "chunk_unknown"),
            "fileType": "video",
            "s3InputTensor": os.environ.get("S3_INPUT_TENSOR"),
            "metadata": json.loads(os.environ.get("METADATA", "{}")) 
        }

        # Run the logic
        result = process_inference_task(event)
        
        logger.info(f"Task completed with status: {result.get('status')}")
        logger.info(f"Result: {json.dumps(result)}")

        if result.get('status') == 'failed':
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.critical(f"Fatal error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()