import config
# from fusion_engine import SoftVotingFusion
import sys


try:
    from . import config
    from .fusion_engine import SoftVotingFusion
except ImportError:
    import config
    from fusion_engine import SoftVotingFusion
# ---------------------------------------------------------


# Initialize the engine once 
engine = SoftVotingFusion(weights=config.FUSION_WEIGHTS)

def _extract_probs_from_response(response_data):
    """
    Parses the specific model output JSON to create a probability distribution.
    Class "0" = Truthful, Class "1" = Deceptive.
    Returns: [Prob_Truthful, Prob_Deceptive]
    """
    if not response_data or 'metadata' not in response_data:
        return None

    meta = response_data['metadata']
    prediction = str(meta.get('prediction', '0')) # "0" or "1"
    confidence = float(meta.get('confidence', 0.5))

    # Normalize into vector: [Prob_Truthful, Prob_Deceptive]
    if prediction == '1':
        return [1.0 - confidence, confidence]
    else:
        return [confidence, 1.0 - confidence]

def determine_primary_modality(video_probs, audio_probs, winning_index, weights):
    """
    Calculates which modality contributed more to the winning label.
    """
    # If Audio is missing, Video is automatically the primary
    if audio_probs is None:
        return "video"

    # Get the probability each modality assigned to the WINNING class
    v_contribution = video_probs[winning_index] * weights.get('video', 0.0)
    a_contribution = audio_probs[winning_index] * weights.get('audio', 0.0)

    # Compare weighted contributions
    if v_contribution >= a_contribution:
        return "video"
    else:
        return "audio"

def process_fusion_request(video_response, audio_response=None):
    try:
        # Extract Inputs
        v_probs = _extract_probs_from_response(video_response)
        a_probs = _extract_probs_from_response(audio_response)

        # Validation
        if audio_response and (video_response.get('chunkId') != audio_response.get('chunkId')):
            return {"status": "error", "message": "Chunk ID mismatch"}

        # Run Inference
        result = engine.predict(video_probs=v_probs, audio_probs=a_probs)

        if result["status"] == "error":
            return result

        # Determine Primary Modality 

        primary = determine_primary_modality(
            video_probs=v_probs, 
            audio_probs=a_probs, 
            winning_index=result['label_index'], 
            weights=config.FUSION_WEIGHTS
        )

        # Format Final Output
        return {
            "primaryModality": primary, # "video" or "audio",
            "prediction": str(result["label_index"]),
            "confidence": result["confidence"]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# # --- Usage ---
# if __name__ == "__main__":    
        
#     video_response = {
#         "sessionId": "session_123",
#         "fileType": "video",
#         "chunkId": "chunk_01",
#         "s3Output": "s3://deceptive-detection-bucket/results",
#         "status": "success",
#         "metadata": { 
#             "originalResolution": {"width": 1920, "height": 1080},
#             "numFrames": 64,
#             "prediction": "0", # Truthful
#             "confidence": 0.60,
#             "durationSeconds": 10 
#         }
#     }

#     audio_response = {
#         "sessionId": "session_123",
#         "fileType": "audio",
#         "chunkId": "chunk_01",
#         "s3Output": "s3://deceptive-detection-bucket/results",
#         "status": "success",
#         "metadata": { 
#             "prediction": "1",      # Deceptive
#             "confidence": 0.95,
#             "durationSeconds": 10 
#         }
#     }

#     print("Running Fusion Process...")
#     final_report = process_fusion_request(video_response, audio_response)

#     print(final_report)