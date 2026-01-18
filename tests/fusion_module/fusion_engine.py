import numpy as np

class SoftVotingFusion:
    def __init__(self, weights):
        self.weights = weights

    def predict(self, video_probs=None, audio_probs=None):
        # Initialize accumulators
        weighted_sum = np.zeros(2) 
        total_weight_used = 0.0
        active_inputs = 0

        # Helper to process inputs safely
        def add_input(probs, modality_key):
            nonlocal weighted_sum, total_weight_used, active_inputs
            
            if probs is not None:
                if len(probs) != 2:
                    raise ValueError(f"Invalid shape for {modality_key}")
                
                w = self.weights.get(modality_key, 0.0)
                weighted_sum += (np.array(probs) * w)
                total_weight_used += w
                active_inputs += 1

        # Aggregate available modalities
        add_input(video_probs, "video")
        add_input(audio_probs, "audio")

        # Handle Edge Case
        if active_inputs == 0 or total_weight_used == 0:
            return {"status": "error", "message": "No valid inputs."}

        # Normalize
        final_probs = weighted_sum / total_weight_used
        
        # Determine Winner
        best_index = int(np.argmax(final_probs))
        confidence = float(final_probs[best_index])

        return {
            "status": "success",
            "label_index": best_index,
            "confidence": confidence,
            "probabilities": final_probs.tolist() 
        }