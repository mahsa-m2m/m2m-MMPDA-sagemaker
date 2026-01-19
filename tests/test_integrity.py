import sys
import os
import json
import torch
import joblib
import numpy as np
import boto3
import tempfile
import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score, classification_report 
from botocore.exceptions import ClientError



LOCAL_CONFIG = {
    "dataset_csv": "test_updated.csv",
    "sample_size": 5,
    
    # Audio Files
    "audio_model_path":  "/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/tests/inference_audio/audio_model.pth",  
    "audio_scaler_path": "/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/tests/inference_audio/scaler.pkl",       
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
    Loads the audio model from local disk
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

def get_label_int(label_str):
    # Assumes: "deceptive" = 1, "truthful" = 0
    clean_label = str(label_str).lower().strip()
    if clean_label == "deceptive":
        return 1
    elif clean_label == "truthful":
        return 0
    else:
        raise ValueError(f"Unknown label: {label_str}")

# MAIN TEST EXECUTION
def run_local_test():
    print("=== STARTING LOCAL INTEGRATION TEST ===\n")

    s3_client = boto3.client('s3')
    local_pt_path = None

    if not os.path.exists(LOCAL_CONFIG["dataset_csv"]):
        print(f"❌ Error: CSV file not found at {LOCAL_CONFIG['dataset_csv']}")
        return
    
    df = pd.read_csv(LOCAL_CONFIG["dataset_csv"])
    print(f"Loaded dataset with {len(df)} samples.")

    desired_sample = LOCAL_CONFIG.get("sample_size")
    if desired_sample and len(df) > desired_sample:
        df = df.sample(n=desired_sample, random_state=42)
        print(f"⚠️ Sampling enabled: Selected {len(df)} random rows out of original dataset.")
    else:
        print(f"Loaded full dataset with {len(df)} samples.")

    # Init Engine
    print("   -> Initializing Video Engine...")
    video_engine = get_inference_engine()

    # Load Model Locally
    audio_model, audio_scaler = load_audio_model_local(
        LOCAL_CONFIG["audio_model_path"], 
        LOCAL_CONFIG["audio_scaler_path"]
    )

    # Initialize Preprocessor
    print("   -> Initializing Audio Preprocessor...")
    audio_extractor = get_feature_extractor()

     # Lists to store results
    y_true = []
    y_pred = []
    results_list = []

    for index, row in df.iterrows():
        s3_url = row['s3_path']
        label_str = row['label']
        
        print(f"\nProcessing [{index+1}/{len(df)}]: {os.path.basename(s3_url)}")
        
        local_pt_path = None
        
        try:
            # A. GROUND TRUTH
            true_label = get_label_int(label_str)

            # B. DOWNLOAD FILE
            bucket, key = parse_s3_path(s3_url)
            try:
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                    s3_client.download_file(bucket, key, tmp.name)
                    local_pt_path = tmp.name
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    # Just skip
                    print(f"   ⚠️ File not found on S3. Skipping.")
                    continue
                else:
                    raise e
            
            # C. VIDEO INFERENCE
            print("---------> video inference")
            video_batch_results = video_engine.predict_batch_silent([local_pt_path], batch_size=video_config.BATCH_SIZE)
            
            if not video_batch_results: raise ValueError("Video engine no results")
            
            v_res = video_batch_results[0]
            v_pred_label = v_res['predicted_label']
            v_conf = v_res['truthful_prob'] if v_pred_label == "0" else v_res['deceptive_prob']

            video_mock = {
                "fileType": "video", "chunkId": "test", "status": "success",
                "metadata": {"prediction": str(v_pred_label), "confidence": float(v_conf)}
            }

            # D. AUDIO INFERENCE
            print("---------> audio inference ")
            pt_data = torch.load(local_pt_path)
            if 'audio_wave' not in pt_data: raise KeyError("Missing 'audio_wave' in .pt")
            
            raw_audio = pt_data['audio_wave']
            audio_feats = audio_extractor.extract(raw_audio)
            
            a_pred_val, a_conf_val = predict_audio(audio_model, audio_scaler, audio_feats)
            
            audio_mock = {
                "fileType": "audio", "chunkId": "test", "status": "success",
                "metadata": {"prediction": str(a_pred_val), "confidence": float(a_conf_val)}
            }

            # E. FUSION
            print("--------- fusion ---------")
            fusion_report = process_fusion_request(video_mock, audio_mock)
            
            if fusion_report.get("status") == "error":
                raise RuntimeError(f"Fusion error: {fusion_report.get('message')}")

            # F. STORE PREDICTION
            final_pred_label = int(fusion_report['prediction']) # "0" or "1"
            
            y_true.append(true_label)
            y_pred.append(final_pred_label)
            
            print(f"   -> Truth: {true_label} | Pred: {final_pred_label} | {fusion_report['primaryModality']} driven")
            
            
            # Get Video Deceptive Probability
            if v_res['predicted_label'] == "1":
                v_prob_deceptive = float(v_res['deceptive_prob'])
            else:
                v_prob_deceptive = 1.0 - float(v_res['truthful_prob'])

            # Get Audio Deceptive Probability
            if a_pred_val == 1:
                a_prob_deceptive = float(a_conf_val)
            else:
                a_prob_deceptive = 1.0 - float(a_conf_val)

            # Store for Analysis
            results_list.append({
                "truth": true_label,
                "video_prob": v_prob_deceptive,
                "audio_prob": a_prob_deceptive
            })

        except Exception as e:
            print(f"   ❌ Failed on file {s3_url}: {e}")
            continue
            
        finally:
            if local_pt_path and os.path.exists(local_pt_path):
                os.unlink(local_pt_path)
    
    # CALCULATE METRICS
    if len(y_true) > 0:
        print("\n" + "="*40)
        print("FINAL EVALUATION RESULTS")
        print("="*40)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary') 
        
        print(f"Total Samples Processed: {len(y_true)}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"F1-Score: {f1:.4f}")
        print("\nDetailed Report:")
        print(classification_report(y_true, y_pred, target_names=["Truthful", "Deceptive"]))
        print("="*40)
    else:
        print("No samples were processed successfully.")
    
    #### To calculate best weight
    # import numpy as np

    # print("\n=== WEIGHT OPTIMIZATION ANALYSIS ===")
    # best_acc = 0
    # best_weights = (0, 0)

    # # Test weights from 0.0 to 1.0
    # for v_weight in np.arange(0.0, 1.1, 0.1):
    #     a_weight = 1.0 - v_weight
        
    #     correct_count = 0
    #     total_count = len(results_list)
        
    #     for item in results_list:
    #         # Soft Vote Formula
    #         final_prob = (item['video_prob'] * v_weight) + (item['audio_prob'] * a_weight)
    #         final_pred = 1 if final_prob > 0.5 else 0
            
    #         if final_pred == item['truth']:
    #             correct_count += 1
                
    #     acc = correct_count / total_count
    #     print(f"Weights [Video: {v_weight:.1f}, Audio: {a_weight:.1f}] -> Accuracy: {acc:.4f}")
        
    #     if acc > best_acc:
    #         best_acc = acc
    #         best_weights = (v_weight, a_weight)

    # print("="*40)
    # print(f"🏆 BEST WEIGHTS: Video={best_weights[0]:.1f}, Audio={best_weights[1]:.1f}")
    # print(f"🚀 POTENTIAL ACCURACY: {best_acc:.4f}")
    # print("="*40)

if __name__ == "__main__":
    run_local_test()