import sys
import os
import torch
import cv2
import numpy as np
import time

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from preprocessing import InferencePreprocessor

# Import the model
try:
    from models_comp.fusion_model import FusionModule, MinimalFusionModule
except ImportError as e:
    print(f"❌ Import Error: {e}")
    raise

from explainer import MultimodalExplainer

class FusionVisualizer:
    def __init__(self):
        print(f"🚀 Initializing Fusion Visualizer on {config.DEVICE}...")
        self.preprocessor = InferencePreprocessor()

        # --- 1. Load Model (Same logic as main.py) ---
        if not hasattr(config.MODEL_ARGS, 'device'):
            config.MODEL_ARGS.device = config.DEVICE
        if not hasattr(config.MODEL_ARGS, 'attn_mask'):
            config.MODEL_ARGS.attn_mask = None
        
        self.model = MinimalFusionModule(config.MODEL_ARGS)
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
            return

        # print("\nDEBUGGING: Printing Face Model Structure")
        # print("------------------------------------------------")
        # print(self.model.face_model)
        # print("------------------------------------------------")
        # # Check what the ResNet part is called. 

        self.model.to(config.DEVICE)
        self.model.eval()
        
        # --- 2. Initialize Explainer ---
        self.explainer = MultimodalExplainer(self.model, target_layer_name='features')
        # self.explainer = MultimodalExplainer(self.model, target_layer_name='layer4')

    def get_raw_video_frames(self, video_path):
        """Helper to read raw video for visualization background"""
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None:
            fps = 25.0

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames), fps

    def visualize_single_video(self, video_path, output_filename='heatmap_output.avi'):
        print(f"🔍 Analyzing: {os.path.basename(video_path)}")
        
        # 1. Preprocess 
        try:
            data = self.preprocessor.process_video(video_path)
            vision_behaviour = data['vision_behaviour'].to(config.DEVICE)
            vision_face = data['vision_face'].to(config.DEVICE)
            audio_mel = data['audio_mel'].to(config.DEVICE)
            audio_wave = data['audio_wave'].to(config.DEVICE)
            
            # Dimension fixes
            if vision_face.dim() == 4: vision_face = vision_face.unsqueeze(0)
            if vision_behaviour.dim() == 2: vision_behaviour = vision_behaviour.unsqueeze(0)
            if audio_mel.dim() == 3: audio_mel = audio_mel.unsqueeze(0)
            if audio_wave.dim() == 1: audio_wave = audio_wave.unsqueeze(0)

        except Exception as e:
            print(f"❌ Preprocessing failed for {video_path}: {e}")
            return

        # ---------------------------------------------------------
        # DEFINING NAMES
        # ---------------------------------------------------------
        defined_names = [
            "L_Eye_Open", "R_Eye_Open", "L_Eye_Width", "R_Eye_Width",
            "Eye_Dist", "L_Upper_Lid", "R_Upper_Lid", "L_Lower_Lid", "R_Lower_Lid",
            "L_Eye_Squint", "R_Eye_Squint", "Eye_Symm",
            "L_Brow_H", "R_Brow_H", "L_Inner_Brow", "R_Inner_Brow",
            "L_Outer_Brow", "R_Outer_Brow", "Brow_Dist", "Brow_Angle",
            "Mouth_Ratio", "Mouth_Width", "Mouth_Height", "Up_Lip_Center", "Low_Lip_Center",
            "L_Corn_H", "R_Corn_H", "Mouth_Open", "Lip_Dist", "Mouth_Asym",
            "Nose_Width", "Nose_Tip_H", "L_Nasolabial", "R_Nasolabial", "Nose_to_Chin",
            "Head_Pitch", "Head_Yaw", "Head_Roll",
            "L_Gaze_H", "L_Gaze_V", "R_Gaze_H", "R_Gaze_V", "Gaze_Conv",
            "Face_Symm", "Face_Width", "Face_Height", "Up_Face_Act", "Low_Face_Act",
            "Valence", "Arousal"
        ]
        
        actual_feats = vision_behaviour.shape[-1]
        if len(defined_names) == actual_feats:
            feature_names = defined_names
        elif actual_feats > len(defined_names):
            extras = [f"Extra_{i}" for i in range(actual_feats - len(defined_names))]
            feature_names = defined_names + extras
        else:
            feature_names = defined_names[:actual_feats]

        # 2. Run Explainer
        inputs = (vision_behaviour, vision_face, audio_mel, audio_wave)
        face_cams, beh_saliency, pred_idx = self.explainer.explain(inputs)
        
        # Label Fix
        pred_label = "Truthful" if pred_idx == 0 else "Deceptive"
        print(f"   Model Prediction: {pred_label}")

        # 3. Generate Video
        self.create_video(video_path, face_cams, beh_saliency, feature_names, output_filename, pred_label)

    def create_video(self, video_path, face_cams, beh_saliency, feature_names, save_path, label):
        """
        Reads video frame-by-frame and writes output immediately.
        Includes Per-Frame Normalization to ensure heatmap is always visible.
        """
        
        # --- 1. Handle Model Output Shapes ---
        if face_cams.ndim == 4 and face_cams.shape[0] == 1:
            face_cams = face_cams.squeeze(0) # (64, 7, 7)
            
        if beh_saliency.ndim == 3 and beh_saliency.shape[0] == 1:
            beh_saliency = beh_saliency.squeeze(0) # (64, 50)

        T_model = face_cams.shape[0]

        # --- 2. Setup Video Streaming ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open source video: {video_path}")
            return

        # Get Metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25.0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # DEBUG: Verify the frame count is logical
        if total_frames <= 0:
            print("⚠️ Warning: Could not determine total frames from header. Progress bar might be inaccurate.")
            total_frames = 100 # Fallback to avoid div/0 errors

        print(f"   🎬 Streaming Video Generation:")
        print(f"      - Source: {total_frames} frames @ {fps:.2f} FPS")
        print(f"      - Model:  {T_model} heatmap frames")

        # Setup Writer
        panel_width = 350
        out_size = (W + panel_width, H)
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter(save_path, fourcc, fps, out_size)
        
        t = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            # --- 3. Time Interpolation ---
            progress = t / total_frames
            t_mod = int(progress * T_model)
            t_mod = min(t_mod, T_model - 1)
            
            # --- 4. Heatmap Overlay ---
            if t_mod < T_model:
                if face_cams.ndim == 4: cam_small = face_cams[t_mod, 0]
                else: cam_small = face_cams[t_mod]
                
                # This ensures every frame has a "max" spot
                c_min, c_max = cam_small.min(), cam_small.max()
                if c_max - c_min > 1e-9:
                    cam_norm = (cam_small - c_min) / (c_max - c_min)
                else:
                    cam_norm = cam_small # Avoid div by zero if flat
                # ===============================================

                cam = cv2.resize(cam_norm, (W, H))
                heatmap = np.uint8(255 * cam)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Apply overlay
                frame = cv2.addWeighted(frame, 0.65, heatmap_colored, 0.35, 0)

            # --- 5. Side Panel ---
            canvas = np.zeros((H, W + panel_width, 3), dtype=np.uint8)
            canvas[:, :W, :] = frame 
            canvas[:, W:, :] = (30, 30, 30)

            cv2.putText(canvas, f"Pred: {label}", (W + 20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # if t_mod < beh_saliency.shape[0]:
            #     scores = beh_saliency[t_mod]
            #     top_indices = np.argsort(scores)[::-1][:6] 
                
            #     y_pos = 80
            #     cv2.putText(canvas, "Top Behaviors:", (W + 20, y_pos), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            #     y_pos += 30

            #     for idx in top_indices:
            #         val = scores[idx]
            #         name = feature_names[idx]
            #         max_val = np.max(scores) + 1e-9 
            #         bar_len = int((val / max_val) * (panel_width - 40))
            #         cv2.rectangle(canvas, (W+20, y_pos-15), (W+20+bar_len, y_pos), (0, 200, 100), -1)
            #         cv2.putText(canvas, f"{name}", (W + 25, y_pos-3), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            #         y_pos += 30

            out.write(canvas)
            t += 1
            
            if t % 50 == 0:
                print(f"      Processed {t}/{total_frames} frames... (Mapped to Model Frame {t_mod})", end='\r')
        
        cap.release()
        out.release()
        print(f"\n✅ Saved explanation to: {save_path}")

if __name__ == "__main__":
    # 1. Initialize
    visualizer = FusionVisualizer()
    
    # 2. A video to test
    test_video = "sample/train/truthful/TTTT_921_class_Truth_19.mp4"
    
    if os.path.exists(test_video):
        visualizer.visualize_single_video(test_video, output_filename="explainability_result.avi")
    else:
        print(f"Video not found: {test_video}")