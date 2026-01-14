import config
from fusion_engine import SoftVotingFusion

# Initialize the engine once 
engine = SoftVotingFusion(weights=config.FUSION_WEIGHTS)

def process_fusion_request(data):
    """
    Main entry point for the pipeline.
    
    Args:
        data (dict): A dictionary containing probabilities.
                     keys: 'video_probs', 'audio_probs'
                     values: list of floats (e.g. [0.9, 0.1]) or None
    
    Returns:
        dict: Final formatted result with label names.
    """
    try:
        # 1. Extract Inputs
        v_in = data.get("video_probs") # Can be None
        a_in = data.get("audio_probs") # Can be None

        # 2. Run Inference
        result = engine.predict(video_probs=v_in, audio_probs=a_in)

        # 3. Check for internal logic errors (e.g., no inputs)
        if result["status"] == "error":
            return result

        # 4. Map Index to String Label (Truthful/Deceptive)
        label_str = config.CLASS_LABELS.get(result["label_index"], "Unknown")

        # 5. Format Final Output
        return {
            "final_label": label_str,
            "confidence": round(result["confidence"], 4),
            "details": {
                "truthful_score": round(result["probabilities"][0], 4),
                "deceptive_score": round(result["probabilities"][1], 4),
                "modalities_used": [
                    k for k, v in [("video", v_in), ("audio", a_in)] if v is not None
                ]
            }
        }

    except Exception as e:
        # Log error here if needed
        return {"status": "error", "message": str(e)}

# --- Usage Example ---
if __name__ == "__main__":
    # Test Case 1: Both Present
    test_input = {
        "video_probs": [0.1, 0.9], # Says Deceptive
        "audio_probs": [0.4, 0.6]  # Says Deceptive
    }
    print("Test 1 (Both):", process_fusion_request(test_input))

    # Test Case 2: Audio Only (Video Missing)
    test_input_2 = {
        "video_probs": None,
        "audio_probs": [0.8, 0.2]  # Says Truthful
    }
    print("Test 2 (Audio Only):", process_fusion_request(test_input_2))
