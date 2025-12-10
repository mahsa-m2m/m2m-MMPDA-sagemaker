import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from preprocessing import InferencePreprocessor

try:
    # from models_comp.fusion_model import MinimalFusionModule
    from models_comp.fusion_model import FusionModule
except ImportError as e:
    print(f"❌ Import Error: {e}")
    raise

# --- HELPER DATASET FOR BATCHING ---
class VideoInferenceDataset(Dataset):
    def __init__(self, video_paths, preprocessor):
        self.video_paths = video_paths
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        try:
            t_start_cpu = time.perf_counter()
            # returns dict of tensors with dim 0 unsqueezed
            data = self.preprocessor.process_video(video_path)
            
            for k, v in data.items():
                data[k] = v.squeeze(0) 
                
            t_end_cpu = time.perf_counter()
            cpu_duration = t_end_cpu - t_start_cpu

            data['video_path'] = video_path # Pass path to track results
            data['valid'] = True
            data['cpu_time'] = cpu_duration 

            return data
        except Exception as e:
            print(f"Skipping {video_path}: {e}")
            # Return a flag indicating failure (handled in collate_fn)
            return {'valid': False, 'video_path': video_path}

def collate_fn_filter_errors(batch):
    """Custom collator to filter out videos that failed preprocessing"""
    batch = [item for item in batch if item['valid']]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class FusionInferenceService:
    def __init__(self):
        print(f"🚀 Initializing Fusion Service on {config.DEVICE}...")
        self.preprocessor = InferencePreprocessor()

        if not hasattr(config.MODEL_ARGS, 'device'):
            config.MODEL_ARGS.device = config.DEVICE
        
        # Also ensure attn_mask exists (it is used in your get_network call)
        if not hasattr(config.MODEL_ARGS, 'attn_mask'):
            config.MODEL_ARGS.attn_mask = None
        
        # self.model = MinimalFusionModule(config.MODEL_ARGS)
        self.model = FusionModule(config.MODEL_ARGS)
        
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

    def predict_single(self, video_path):
        """Legacy method for single video"""
        # (Your existing code here...)
        pass

    def predict_batch(self, video_paths, batch_size=8, num_workers=4):
        """
        Scalable method for list of videos.
        Args:
            video_paths: List of file paths
            batch_size: How many videos to push to GPU at once (Try 8, 16, 32)
            num_workers: CPU cores for parallel preprocessing
        """
        dataset = VideoInferenceDataset(video_paths, self.preprocessor)
        
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
        print(f"🔄 Processing {len(video_paths)} videos in batches of {batch_size}...")

        total_script_start = time.perf_counter()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                if batch_data is None: continue # Skip empty batches
                
                cpu_times = batch_data['cpu_time'].numpy().tolist() 
                paths = batch_data['video_path']

                # 1. Move Batch to GPU
                vision_behaviour = batch_data['vision_behaviour'].to(config.DEVICE)
                vision_face = batch_data['vision_face'].to(config.DEVICE)
                audio_mel = batch_data['audio_mel'].to(config.DEVICE)
                audio_wave = batch_data['audio_wave'].to(config.DEVICE)
                current_batch_len = len(paths)
                paths = batch_data['video_path']

                if config.DEVICE == 'cuda': torch.cuda.synchronize()
                t_gpu_start = time.perf_counter()

                # 2. Batch Inference (GPU processes N videos at once)
                outputs = self.model(
                    vision_behaviour=vision_behaviour,
                    vision_face=vision_face,
                    audio_mel=audio_mel,
                    audio_wave=audio_wave
                )

                if config.DEVICE == 'cuda': torch.cuda.synchronize()
                t_gpu_end = time.perf_counter()

                # Calculate GPU time per video (Amortized)
                # We divide the total batch time by the number of videos in the batch
                total_batch_gpu_time = t_gpu_end - t_gpu_start
                avg_gpu_per_video = total_batch_gpu_time / current_batch_len
                
                
                logits = outputs[0] # [Batch_Size, Num_Classes]
                probs = F.softmax(logits, dim=1).cpu().numpy()

                # 3. Collect Results
                for i, path in enumerate(paths):
                    p_list = probs[i].tolist()

                    # Individual CPU time + Average GPU time
                    video_cpu_time = cpu_times[i]
                    video_total_time = video_cpu_time + avg_gpu_per_video

                    results.append({
                        "video_path": path,
                        "truthful_prob": p_list[0],
                        "deceptive_prob": p_list[1],
                        "predicted_label": "Deceptive" if p_list[1] > p_list[0] else "Truthful",
                        # TIMING DATA
                        "time_cpu": video_cpu_time,
                        "time_gpu": avg_gpu_per_video,
                        "time_total": video_total_time
                    })
        total_script_end = time.perf_counter()
        print(f"🏁 Total Wall-Clock Time for all videos: {total_script_end - total_script_start:.2f}s")

        return results

if __name__ == "__main__":
    service = FusionInferenceService()
    
    # Example: List of 100 videos
    video_list = ["/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/sample/W_453_class_Truth_301.mkv", "/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/sample/TTTT_433_class_Truth_67.mkv"] 
    
    # Run in batch mode
    batch_results = service.predict_batch(video_list, batch_size=4, num_workers=4)
    print(f"Processed {len(batch_results)} videos.")

    # Print formatted results
    print(f"{'VIDEO NAME':<40} | {'LABEL':<10} | {'CPU(s)':<8} | {'GPU(s)':<8} | {'TOTAL(s)':<8}")
    print("-" * 90)

    for res in batch_results:
        vid_name = os.path.basename(res['video_path'])
        print(f"{vid_name:<40} | {res['predicted_label']:<10} | "
              f"{res['time_cpu']:.4f}   | {res['time_gpu']:.4f}   | {res['time_total']:.4f}")

    # for res in batch_results:
    #     print(f"[{res['predicted_label']}] {os.path.basename(res['video_path'])} "
    #           f"(Deceptive: {res['deceptive_prob']:.4f}, Truthful: {res['truthful_prob']:.4f})")