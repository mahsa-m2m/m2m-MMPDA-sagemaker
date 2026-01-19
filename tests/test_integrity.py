import sys
import os
import json
import torch
import joblib
import numpy as np
import boto3
import tempfile


LOCAL_CONFIG = {
    
    "s3_input_pt_url": "s3://deception-detection-bucket/dataset/video/precomputed_features/test/BF006_3NT.pt", 
    # Audio Files
    "audio_model_path":  "/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/tests/inference_audio/audio_model.pth",  
    "audio_scaler_path": "/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/tests/inference_audio/scaler.pkl",       
    "audio_input_json":  "embeddings.json",
    
    # Video Files
    "video_input_pt":    "s3://deception-detection-bucket/test-session-01/video/chunk_001/feature.pt"    
    # "use_same_pt_for_video": True
}

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'inference_audio'))
sys.path.append(os.path.join(current_dir, 'inference_video'))
sys.path.append(os.path.join(current_dir, 'fusion_module'))
sys.path.append(os.path.join(current_dir, 'preprocessor_audio')) 


from preprocessor_audio.preprocessor import AudioFeatureExtractor, get_feature_extractor

from inference_audio.main import BiLSTMAttentionModel, predict as predict_audio

from inference_video.main import get_inference_engine
import inference_video.config as video_config


from fusion_module.main import process_fusion_request

def parse_s3_path(s3_path: str):
    parts = s3_path.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]

# LOAD AUDIO MODEL LOCALLY
def load_audio_model_local(model_path, scaler_path):
    """
    Loads the audio model from local disk, bypassing S3.
    """
    print(f"   -> Loading Audio weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Audio model not found at {model_path}")

    model = BiLSTMAttentionModel(
        input_size=768,
        hidden_size=128,
        num_layers=2,
        num_heads=8,
        dropout=0.5
    )
    
    # Load Weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load Scaler
    print(f"   -> Loading Scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    return model, scaler

# MAIN TEST EXECUTION
def run_local_test():
    print("=== STARTING LOCAL INTEGRATION TEST ===\n")

    s3_client = boto3.client('s3')
    local_pt_path = None

    try:
        print("--- Downloading Input Data ---")
        bucket, key = parse_s3_path(LOCAL_CONFIG["video_input_pt"])
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            print(f"   -> Downloading from {LOCAL_CONFIG['video_input_pt']}...")
            s3_client.download_file(bucket, key, tmp.name)
            local_pt_path = tmp.name
        
        print(f"   -> Downloaded to: {local_pt_path}")

        pt_data = torch.load(local_pt_path)
        print(f"   -> Keys found in .pt: {list(pt_data.keys())}")
    except Exception as e:
            print(f"\n❌ Test Failed: {e}")

    ######### VIDEO INFERENCE
    print("\n--- [2] Running Video Module (Local) ---")
    video_response_mock = None
    try:
        # Init Engine
        print("   -> Initializing Video Engine...")
        video_engine = get_inference_engine()

        # Run Prediction on Local File
        video_path = local_pt_path #LOCAL_CONFIG["video_input_pt"]
        print(f"   -> Processing File: {video_path}")
        # print(video_path)
        # if not os.path.exists(video_path):
        #     raise FileNotFoundError(f"Video input not found at {video_path}")

        batch_results = video_engine.predict_batch_silent(
            [video_path], 
            batch_size=video_config.BATCH_SIZE
        )

        if not batch_results:
            raise ValueError("Video engine returned no results.")

        # Parse Result
        v_result = batch_results[0]
        pred_label = v_result['predicted_label']
        
        # Calculate single confidence value
        if pred_label == "0":
            final_conf = v_result['truthful_prob']
        else:
            final_conf = v_result['deceptive_prob']

        print(f"   ✅ Video Result: Class {pred_label} | Conf {final_conf:.4f}")

        # Create Mock Response
        video_response_mock = {
            "fileType": "video",
            "chunkId": "local_test_chunk",
            "status": "success",
            "metadata": {
                "prediction": str(pred_label),
                "confidence": float(final_conf)
            }
        }

    except Exception as e:
        print(f"   ❌ Video Failed: {e}")
        return

    ######### AUDIO INFERENCE
    print("--- [1] Running Audio Module (Local) ---")
    audio_response_mock = None
    try:
        # Load Model Locally
        model, scaler = load_audio_model_local(
            LOCAL_CONFIG["audio_model_path"], 
            LOCAL_CONFIG["audio_scaler_path"]
        )

        # PREPROCESSING: Load .pt file and Extract Features
        # We use the same .pt file defined in LOCAL_CONFIG["video_input_pt"]
        # pt_file_path = LOCAL_CONFIG["video_input_pt"]
        print(f"   -> Reading Input .pt file: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input file not found at {video_path}")

        # Load dictionary from .pt
        pt_data = torch.load(video_path)
        
        if 'audio_wave' not in pt_data:
            raise KeyError(f"The file {video_path} does not contain the key 'audio_wave'")

        raw_audio_tensor = pt_data['audio_wave']
        print(f"   -> Found raw audio tensor: {raw_audio_tensor.shape}")

        # Initialize Preprocessor
        print("   -> Initializing Audio Preprocessor...")
        extractor = get_feature_extractor()
        
        # Extract features (Pass the tensor directly)
        features = extractor.extract(raw_audio_tensor)
        print(f"   -> Features Extracted. Shape: {features.shape}")

        # # Read Local Input File
        # print(f"   -> Reading Input: {LOCAL_CONFIG['audio_input_json']}")
        # with open(LOCAL_CONFIG["audio_input_json"], 'r') as f:
        #     data = json.load(f)
        #     features = np.array(data['features'])
        #     print(features.shape)

        # Predict
        prediction, confidence = predict_audio(model, scaler, features)
        print(f"   ✅ Audio Result: Class {prediction} | Conf {confidence:.4f}")

        # Create Mock Response
        audio_response_mock = {
            "fileType": "audio",
            "chunkId": "local_test_chunk",
            "status": "success",
            "metadata": {
                "prediction": str(prediction),
                "confidence": float(confidence)
            }
        }

    except Exception as e:
        print(f"   ❌ Audio Failed: {e}")
        return
    
    
    ######## FUSION
    print("\n--- [3] Running Fusion Module (Local) ---")
    try:
        final_report = process_fusion_request(
            video_response_mock, 
            audio_response_mock
        )

        if final_report.get("status") == "error":
            print(f"   ❌ Fusion Error: {final_report.get('message')}")
        else:
            print("\n" + "="*40)
            print("FINAL LOCAL RESULT")
            print("="*40)
            print(json.dumps(final_report, indent=4))
            print("="*40)

    except Exception as e:
        print(f"   ❌ Fusion Logic Failed: {e}")

if __name__ == "__main__":
    run_local_test()