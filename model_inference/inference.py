import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time

import json
import boto3
import tempfile
from typing import Dict, Any


# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)


try:
    from models_comp.fusion_model import MinimalFusionModule, FusionModule, FusionModuleSilent
    import config

except ImportError as e:
    print(f"❌ Import Error: {e}")
    raise



class VideoInferenceError(Exception):
    """Raised when inference fails."""
    pass

class S3WriteError(Exception):
    """Raised when uploading results to S3 fails."""
    pass

class InvalidInputError(Exception):
    """Raised when input parameters or files are missing/invalid."""
    pass


# --- HELPER DATASET FOR BATCHING ---
class VideoInferenceDataset(Dataset):
    def __init__(self, feature_paths):
        """
        Args:
            feature_paths: List of paths to .pt files containing pre-extracted features
            {
                'vision_behaviour': vision_behaviour.unsqueeze(0),  # [1, N, 50]
                'vision_face': vision_face.unsqueeze(0),            # [1, 3, N, H, W]
                'audio_mel': audio_mel.unsqueeze(0),                # [1, 3, N_MELS, T]
                'audio_wave': audio_wave.unsqueeze(0)               # [1, AUDIO_LENGTH]
            }
        """
        self.feature_paths = feature_paths

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):

        feature_path = self.feature_paths[idx]
        try:

            data = torch.load(feature_path, map_location=config.DEVICE)
 
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.squeeze(0) 
            
            if 'vision_face' in data:
                vision_face = data['vision_face']
                
                # 1. Convert from uint8 to float32 [0-1]
                vision_face = vision_face.float() / 255.0
                
                # 2. Normalize
                # Shape: [3, T, H, W]
                mean = torch.tensor([0.485, 0.456, 0.406], device=config.DEVICE).view(3, 1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=config.DEVICE).view(3, 1, 1, 1)
                vision_face = (vision_face - mean) / std
                
                data['vision_face'] = vision_face

            # Store feature path 
            data['feature_path'] = feature_path
            data['valid'] = True

            return data
        except Exception as e:
            print(f"Skipping {feature_path}: {e}")
            return {'valid': False, 'feature_path': feature_path}

def collate_fn_filter_errors(batch):
    """Custom collator to filter out videos that failed preprocessing"""
    batch = [item for item in batch if item['valid']]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class FusionInference:
    def __init__(self):
        print(f"🚀 Initializing Fusion Service on {config.DEVICE}...")

        if not hasattr(config.MODEL_ARGS, 'device'):
            config.MODEL_ARGS.device = config.DEVICE
        
        # Also ensure attn_mask exists (it is used in the get_network call)
        if not hasattr(config.MODEL_ARGS, 'attn_mask'):
            config.MODEL_ARGS.attn_mask = None
        
        self.model = FusionModuleSilent(config.MODEL_ARGS)
        # self.model = MinimalFusionModule(config.MODEL_ARGS)
        # self.model = FusionModule(config.MODEL_ARGS)
        
        try:
            checkpoint = torch.load(config.MODEL_WEIGHTS_PATH, map_location=config.DEVICE)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Weights loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")

        self.model.to(config.DEVICE)
        self.model.eval()

    def predict_batch(self, feature_paths, batch_size=8, num_workers=4):
        """
        Scalable method for list of pre-extracted features.
        Args:
            feature_paths: List of .pt file paths containing pre-extracted features
            batch_size: How many features to push to GPU at once
            num_workers: CPU cores for parallel loading
        """
        dataset = VideoInferenceDataset(feature_paths)
        
        if config.DEVICE == 'cuda':
            torch.cuda.synchronize()

        
        # DataLoader handles parallel CPU processing
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=collate_fn_filter_errors
        )

        results = []
        print(f"🔄 Processing {len(feature_paths)} pre-extracted features in batches of {batch_size}...")

        # total_script_start = time.perf_counter()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                if batch_data is None: continue # Skip empty batches
                
                # cpu_times = batch_data['cpu_time'].numpy().tolist() 
                paths = batch_data['feature_path']

                # 1. Move Batch to GPU
                vision_behaviour = batch_data['vision_behaviour'].to(config.DEVICE)
                vision_face = batch_data['vision_face'].to(config.DEVICE)
                audio_mel = batch_data['audio_mel'].to(config.DEVICE)
                audio_wave = batch_data['audio_wave'].to(config.DEVICE)
                current_batch_len = len(paths)
                paths = batch_data['video_path']

                # if config.DEVICE == 'cuda': torch.cuda.synchronize()
                # t_gpu_start = time.perf_counter()

                # 2. Batch Inference (GPU processes N videos at once)
                outputs = self.model(
                    vision_behaviour=vision_behaviour,
                    vision_face=vision_face,
                    audio_mel=audio_mel,
                    audio_wave=audio_wave
                )
                
                logits = outputs[0] # [Batch_Size, Num_Classes]
                probs = F.softmax(logits, dim=1).cpu().numpy()

                # 3. Collect Results
                for i, path in enumerate(paths):
                    p_list = probs[i].tolist()

                    # # Individual CPU time + Average GPU time
                    # video_cpu_time = cpu_times[i]
                    # video_total_time = video_cpu_time + avg_gpu_per_video

                    results.append({
                        "feature_path": path,
                        "truthful_prob": p_list[0],
                        "deceptive_prob": p_list[1],
                        "predicted_label": "Deceptive" if p_list[1] > p_list[0] else "Truthful"
                    })

        return results

    def predict_batch_silent(self, feature_paths, batch_size=4, num_workers=0):
        """
        Using only Vision + Face features
        (Silent Mode: Audio is ignored).
        Args:
            feature_paths: List of .pt file paths containing pre-extracted features
        """
        # Pass feature_paths
        dataset = VideoInferenceDataset(feature_paths)

        if config.DEVICE == 'cuda':
            torch.cuda.synchronize()
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=collate_fn_filter_errors
        )

        results = []
        print(f"🔄 Processing {len(feature_paths)} pre-extracted features (Silent Mode)...")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                if batch_data is None: continue # Skip empty batches
                # 1. Load Visual and Face features
                vision_behaviour = batch_data['vision_behaviour'].to(config.DEVICE)
                vision_face = batch_data['vision_face'].to(config.DEVICE)

                paths = batch_data['feature_path']

                # 2. Forward pass
                outputs = self.model(
                    vision_behaviour, 
                    vision_face, 
                    audio_mel=None, 
                    audio_wave=None
                )

                # Handle model return type
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    fused_logit = outputs[0]
                else:
                    fused_logit = outputs

                # 3. Calculate Probabilities
                probs = F.softmax(fused_logit, dim=1).cpu().numpy()

                # 4. Format Results
                for i, path in enumerate(paths):
                    p_list = probs[i].tolist()
                    
                    results.append({
                        "feature_path": path,
                        "truthful_prob": p_list[0],
                        "deceptive_prob": p_list[1],
                        "predicted_label": "1" if p_list[1] > p_list[0] else "0" # 1: Deceptive  # 0: Truthful
                    })

        return results


"""
    INPUT:
    {
    "sessionId": "string",
    "fileType": "video",
    "chunkId": "chunk_01",
    "s3InputTensor": "s3://deception-results/session_123/video/chunk_01/preprocessed.json",
    "modelVersion": "video-model-v1.0",
    "metadata": {
        "originalResolution": {"width": 1920, "height": 1080},
        "numFrames": 120
    }
    }

    OUTPUT:
    {
    "sessionId": "string",
    "fileType": "video",
    "chunkId": "chunk_01",
    "s3Output": "s3://deception-results/session_123/video/chunk_01/inference.json",
    "status": "success",
    "metadata": {
        "confidence": 0.92,
        "prediction": 1,
        "numFrames": 120,
        "durationSeconds": 10,
        "modelVersion": "video-model-v1.0"
    },
    "error": null
}
"""

s3_client = boto3.client('s3')
inference_engine = None

def get_inference_engine():
    global inference_engine
    if inference_engine is None:
        inference_engine = FusionInference()
    return inference_engine

def parse_s3_path(s3_path: str) -> tuple:
    parts = s3_path.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]

def lambda_handler(event: Dict[str, Any], context=None) -> Dict[str, Any]:

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
        required_keys = ['sessionId', 'chunkId', 's3InputTensor']
        for key in required_keys:
            if key not in event:
                raise KeyError(key)

        input_tensor_s3_url = event['s3InputTensor']
        session_id = event['sessionId']
        chunk_id = event['chunkId']
        
        input_bucket, input_key = parse_s3_path(input_tensor_s3_url)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            s3_client.download_file(input_bucket, input_key, tmp_file.name)
            chunk_path = tmp_file.name
        
        model = get_inference_engine()

        batch_results = model.predict_batch_silent(
            [chunk_path], 
            batch_size=config.BATCH_SIZE
        )
        # print('========================')
        # print(batch_results)

        if not batch_results:
            raise ValueError("Model inference returned no results.")

        output_key = f"results/{session_id}/video/{chunk_id}/inference.json"

        # Success Response
        response_template['status'] = 'success' 
        response_template['s3Output'] = f"s3://{input_bucket}/{output_key}"
        response_template['metadata']['prediction'] = batch_results[0]['predicted_label']
        
        if batch_results[0]['predicted_label'] == "0":
            response_template['metadata']['confidence'] = batch_results[0]['truthful_prob']
        else:
            response_template['metadata']['confidence'] = batch_results[0]['deceptive_prob']

        try:
            s3_client.put_object(
                Bucket=input_bucket,
                Key=output_key,
                Body=json.dumps(response_template),
            )  
        except Exception as e:
            raise S3WriteError(f"Failed to write results: {str(e)}")

        return response_template  
    
    except KeyError as e:
        response_template['error'] = f"Missing required field: {str(e)}"
        return response_template

    except S3WriteError as e:
        response_template['error'] = str(e)
        return response_template

    except Exception as e:
        print(f"Server Error: {str(e)}")
        response_template['error'] = f"InferenceError: {str(e)}"
        return response_template

    finally:
        # Cleanup temp file
        if chunk_path and os.path.exists(chunk_path):
            os.unlink(chunk_path)



# if __name__ == "__main__":
#     service = FusionInference()
    
#     # Example: List of 100 features.pt
#     feature_list = ["/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/model_preprocessor/test.pt"]
    
#     # Run in batch mode
#     batch_results = service.predict_batch_silent(feature_list, batch_size=4, num_workers=4)

#     print(f"Processed {len(batch_results)} videos.")

#     # Print formatted results
#     print(f"{'VIDEO NAME':<40} | {'LABEL':<10} | {'DECEPTIVE SCORE':<8} | {'TRUTHFUL SCORE':<8}")
#     print("-" * 90)

#     for res in batch_results:
#         vid_name = os.path.basename(res['feature_path'])

#         print(f"{vid_name:<40} | {res['predicted_label']:<10} | "
#               f"{res['deceptive_prob']:.4f}          |  {res['truthful_prob']:.4f} ")
