import numpy as np

class SoftVotingFusion:
    def __init__(self, weights):
        """
        Args:
            weights (dict): e.g., {'video': 0.4, 'audio': 0.6}
        """
        self.weights = weights

    def predict(self, video_probs=None, audio_probs=None):
        """
        Performs weighted soft voting.
        
        Args:
            video_probs (list or np.array): [Prob_Truthful, Prob_Deceptive] OR None
            audio_probs (list or np.array): [Prob_Truthful, Prob_Deceptive] OR None
            
        Returns:
            dict: Result object with label, confidence, and raw probs.
        """
        
        # 1. Initialize accumulators
        weighted_sum = np.zeros(2)  # [0.0, 0.0]
        total_weight_used = 0.0
        active_inputs = 0

        # 2. Helper to process inputs safely
        def add_input(probs, modality_key):
            nonlocal weighted_sum, total_weight_used, active_inputs
            
            if probs is not None:
                # Validation: Ensure we got 2 float values
                if len(probs) != 2:
                    raise ValueError(f"Invalid shape for {modality_key}: expected 2 probs, got {len(probs)}")
                
                w = self.weights.get(modality_key, 0.0)
                weighted_sum += (np.array(probs) * w)
                total_weight_used += w
                active_inputs += 1

        # 3. Aggregate available modalities
        add_input(video_probs, "video")
        add_input(audio_probs, "audio")

        # 4. Handle Edge Case: No valid inputs
        if active_inputs == 0 or total_weight_used == 0:
            return {
                "status": "error",
                "message": "No valid video or audio probabilities provided."
            }

        # 5. Normalize (The Logic: Weighted_Sum / Sum_of_Active_Weights)
        final_probs = weighted_sum / total_weight_used
        
        # 6. Determine Winner
        best_index = int(np.argmax(final_probs))
        confidence = float(final_probs[best_index])

        return {
            "status": "success",
            "label_index": best_index,
            "confidence": confidence,
            "probabilities": final_probs.tolist() # [Prob_Truthful, Prob_Deceptive]
        }