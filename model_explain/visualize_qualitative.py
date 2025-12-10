import sys
import os
import torch
import cv2
import numpy as np
import time

# --- PROJECT SETUP (Matches your main.py) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from preprocessing import InferencePreprocessor

# Import your model
try:
    # from models_comp.fusion_model import MinimalFusionModule
    from models_comp.fusion_model import FusionModule
except ImportError as e:
    print(f"❌ Import Error: {e}")
    raise

# Import the Explainer Logic (From the previous step)
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
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)

    # def visualize_single_video(self, video_path, output_filename='heatmap_output.avi'):
    #     print(f"🔍 Analyzing: {os.path.basename(video_path)}")
        
    #     # 1. Preprocess Data (Get Tensors)
    #     try:
    #         # data dict contains tensors usually with shape (1, T, ...) or (C, T, ...)
    #         data = self.preprocessor.process_video(video_path)
            
    #         # Ensure batch dim exists (1, ...)
    #         # Adjust based on what your preprocessor returns. 
    #         # If it returns (C, T, H, W), we need to unsqueeze(0).
    #         # Based on your Dataset code, it seems your preprocessor returns unsqueezed data?
    #         # Let's verify standard shape:
    #         vision_behaviour = data['vision_behaviour'].to(config.DEVICE)
    #         vision_face = data['vision_face'].to(config.DEVICE)
    #         audio_mel = data['audio_mel'].to(config.DEVICE)
    #         audio_wave = data['audio_wave'].to(config.DEVICE)
            
    #         # Sanity check dimensions
    #         if vision_face.dim() == 4: # (T, C, H, W) -> Make it (1, T, C, H, W)
    #              vision_face = vision_face.unsqueeze(0)
    #         if vision_behaviour.dim() == 2: # (T, F) -> Make it (1, T, F)
    #              vision_behaviour = vision_behaviour.unsqueeze(0)
    #         if audio_mel.dim() == 3: 
    #              audio_mel = audio_mel.unsqueeze(0)
    #         if audio_wave.dim() == 1:
    #              audio_wave = audio_wave.unsqueeze(0)

    #     except Exception as e:
    #         print(f"❌ Preprocessing failed for {video_path}: {e}")
    #         return

    #     # 2. Get Raw Video for Background
    #     raw_frames = self.get_raw_video_frames(video_path)
    #     if len(raw_frames) == 0:
    #         print("❌ Could not read raw video frames.")
    #         return


    #     feature_names = [
    #     # --- EYES (0-11) ---
    #     "L_Eye_Open", "R_Eye_Open", "L_Eye_Width", "R_Eye_Width",
    #     "Eye_Dist", "L_Upper_Lid", "R_Upper_Lid", "L_Lower_Lid", "R_Lower_Lid",
    #     "L_Eye_Squint", "R_Eye_Squint", "Eye_Symm",

    #     # --- BROWS (12-19) ---
    #     "L_Brow_H", "R_Brow_H", "L_Inner_Brow", "R_Inner_Brow",
    #     "L_Outer_Brow", "R_Outer_Brow", "Brow_Dist", "Brow_Angle",

    #     # --- MOUTH (20-29) ---
    #     "Mouth_Ratio", "Mouth_Width", "Mouth_Height", "Up_Lip_Center", "Low_Lip_Center",
    #     "L_Corn_H", "R_Corn_H", "Mouth_Open", "Lip_Dist", "Mouth_Asym",

    #     # --- NOSE/CHEEK (30-34) ---
    #     "Nose_Width", "Nose_Tip_H", "L_Nasolabial", "R_Nasolabial", "Nose_to_Chin",

    #     # --- GAZE/POSE (35-42) ---
    #     "Head_Pitch", "Head_Yaw", "Head_Roll",
    #     "L_Gaze_H", "L_Gaze_V", "R_Gaze_H", "R_Gaze_V", "Gaze_Conv",

    #     # --- EXPRESSION (43-47) ---
    #     "Face_Symm", "Face_Width", "Face_Height", "Up_Face_Act", "Low_Face_Act",

    #     # --- EMOTION (48-49) ---
    #     "Valence", "Arousal"
    #     ]
    #     # If the model input isn't exactly 50, prevent a crash
    #     actual_feats = vision_behaviour.shape[-1]
    #     if len(feature_names) != actual_feats:
    #         print(f"⚠️ WARNING: Feature mismatch! Model expects {actual_feats}, list has {len(feature_names)}.")
    #         # Fallback to dummy names so code continues running
    #         feature_names = [f"Feat_{i}" for i in range(actual_feats)]

    #     # 3. Run Explainer
    #     inputs = (vision_behaviour, vision_face, audio_mel, audio_wave)
        
    #     # This returns the Face Heatmap and Behavioral Saliency
    #     face_cams, beh_saliency, pred_idx = self.explainer.explain(inputs)
        
    #     pred_label = "Deceptive" if pred_idx == 0 else "Truthful" # Adjust index mapping if needed
    #     print(f"   Model Prediction: {pred_label}")

    #     # 4. Generate Video Overlay
    #     # Create Dummy Feature Names (or map them if you know your specific OpenFace columns)
    #     num_feats = vision_behaviour.shape[-1]
    #     feature_names = [f"Feat_{i}" for i in range(num_feats)]
    #     # Example mapping if you know them:
    #     # feature_names = ['GazeX', 'GazeY', 'AU1', 'AU2', ...] 

    #     self.create_video(raw_frames, face_cams, beh_saliency, feature_names, output_filename, pred_label)

    def visualize_single_video(self, video_path, output_filename='heatmap_output.avi'):
        print(f"🔍 Analyzing: {os.path.basename(video_path)}")
        
        try:
            # 1. Preprocess
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

        # 2. Get Raw Video
        raw_frames = self.get_raw_video_frames(video_path)
        if len(raw_frames) == 0: return

        # ---------------------------------------------------------
        # DEFINING NAMES (Standard 50 from MediaPipe)
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

        # ---------------------------------------------------------
        # ROBUST MERGE (The Fix)
        # ---------------------------------------------------------
        actual_feats = vision_behaviour.shape[-1]
        
        # If we have the exact number, great.
        if len(defined_names) == actual_feats:
            feature_names = defined_names
        # If model has MORE features than names, use names and label the rest "Extra"
        elif actual_feats > len(defined_names):
            print(f"⚠️ Note: Model has {actual_feats} features, but we only have names for {len(defined_names)}.")
            extras = [f"Extra_{i}" for i in range(actual_feats - len(defined_names))]
            feature_names = defined_names + extras
        # If model has FEWER features, just take the first N names
        else:
            print(f"⚠️ Note: Model has {actual_feats} features. Truncating name list.")
            feature_names = defined_names[:actual_feats]
            
        # ---------------------------------------------------------

        # 3. Run Explainer
        inputs = (vision_behaviour, vision_face, audio_mel, audio_wave)
        face_cams, beh_saliency, pred_idx = self.explainer.explain(inputs)
        
        pred_label = "Deceptive" if pred_idx == 0 else "Truthful"
        print(f"   Model Prediction: {pred_label}")

        # 4. Generate Video
        self.create_video(raw_frames, face_cams, beh_saliency, feature_names, output_filename, pred_label)

    def create_video(self, raw_frames, face_cams, beh_saliency, feature_names, save_path, label):
        """Combine raw video, heatmap, and behavior charts"""
        T_raw, H, W, _ = raw_frames.shape
        T_model = face_cams.shape[0]
        
        # Setup Video Writer
        panel_width = 350
        out_size = (W + panel_width, H)
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Or 'mp4v'
        out = cv2.VideoWriter(save_path, fourcc, 25.0, out_size)
        
        # Beh Saliency is (Time, Feats). squeeze batch if needed.
        if len(beh_saliency.shape) == 3: 
            beh_saliency = beh_saliency[0]

        print(f"   Generating video ({T_raw} frames)...")

        for t in range(T_raw):
            frame = raw_frames[t].copy()
            
            # --- A. Face Heatmap ---
            # Map raw time 't' to model time
            # Simple approach: If model downsampled video, clamp index
            t_mod = min(t, T_model - 1) 
            
            if t_mod < T_model:
                # Resize 7x7 CAM to Video Size
                cam = cv2.resize(face_cams[t_mod, 0], (W, H))
                heatmap = np.uint8(255 * cam)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Overlay
                frame = cv2.addWeighted(frame, 0.65, heatmap_colored, 0.35, 0)

            # --- B. Side Panel (Behavior) ---
            canvas = np.zeros((H, W + panel_width, 3), dtype=np.uint8)
            canvas[:, :W, :] = frame # Put face on left
            
            # Background for panel
            canvas[:, W:, :] = (30, 30, 30) # Dark gray

            # Header
            cv2.putText(canvas, f"Pred: {label}", (W + 20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Top Behaviors
            if t_mod < beh_saliency.shape[0]:
                scores = beh_saliency[t_mod]
                top_indices = np.argsort(scores)[::-1][:6] # Top 6
                
                y_pos = 80
                cv2.putText(canvas, "Top Behaviors (Impact):", (W + 20, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 30

                for idx in top_indices:
                    val = scores[idx]
                    name = feature_names[idx]
                    
                    # Normalize bar length roughly
                    # bar_len = int(val * 2000) # Scaling factor might need tuning based on gradients
                    # bar_len = min(bar_len, panel_width - 40)
                    max_val_in_frame = np.max(scores) + 1e-9 # Avoid div by zero
                    relative_val = val / max_val_in_frame    # 0.0 to 1.0
                    bar_len = int(relative_val * (panel_width - 40))
                    
                    # Draw Bar
                    cv2.rectangle(canvas, (W+20, y_pos-15), (W+20+bar_len, y_pos), (0, 200, 100), -1)
                    # Draw Name
                    cv2.putText(canvas, f"{name}", (W + 25, y_pos-3), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    y_pos += 30

            out.write(canvas)
        
        out.release()
        print(f"✅ Saved explanation to: {save_path}")

if __name__ == "__main__":
    # 1. Initialize
    visualizer = FusionVisualizer()
    
    # 2. A video to test
    test_video = "/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/sample/train/deceptive/Adele lying.mp4"
    
    if os.path.exists(test_video):
        visualizer.visualize_single_video(test_video, output_filename="explainability_result.avi")
    else:
        print(f"Video not found: {test_video}")