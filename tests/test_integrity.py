import sys
import os
import json
import torch
import joblib
import numpy as np


LOCAL_CONFIG = {
    # Audio Files
    "audio_model_path":  "/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/tests/inference_audio/audio_model.pth",  
    "audio_scaler_path": "/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/tests/inference_audio/scaler.pkl",       
    "audio_input_json":  "embeddings.json",
    
    # Video Files
    "video_input_pt":    "/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/tests/inference_video/feature.pt"    
}

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'inference_audio'))
sys.path.append(os.path.join(current_dir, 'inference_video'))
sys.path.append(os.path.join(current_dir, 'fusion_module'))

from inference_audio.main import BiLSTMAttentionModel, predict as predict_audio

from inference_video.main import get_inference_engine
import inference_video.config as video_config


from fusion_module.main import process_fusion_request

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


    # VIDEO INFERENCE
    print("\n--- [2] Running Video Module (Local) ---")
    video_response_mock = None
    try:
        # Init Engine
        print("   -> Initializing Video Engine...")
        video_engine = get_inference_engine()

        # Run Prediction on Local File
        video_path = LOCAL_CONFIG["video_input_pt"]
        print(f"   -> Processing File: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video input not found at {video_path}")

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

    # AUDIO INFERENCE
    print("--- [1] Running Audio Module (Local) ---")
    audio_response_mock = None
    try:
        # Load Model Locally
        model, scaler = load_audio_model_local(
            LOCAL_CONFIG["audio_model_path"], 
            LOCAL_CONFIG["audio_scaler_path"]
        )

        # Read Local Input File
        print(f"   -> Reading Input: {LOCAL_CONFIG['audio_input_json']}")
        with open(LOCAL_CONFIG["audio_input_json"], 'r') as f:
            data = json.load(f)
            features = np.array(data['features'])
            print(features.shape)

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
    # FUSION
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