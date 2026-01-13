import json
import datetime

class DeceptionReportGenerator:
    def __init__(self, case_id, file_name):
        self.case_id = case_id
        self.file_name = file_name
        self.chunks = []
        self.modality = "Unknown"

    def add_chunk(self, api_response):
        """
        Parses the input.
        
        :param api_response: Dict containing metadata (sessionId, chunkId, etc.)
        :param inference_result: List containing the inference dict (probs, label)

        """
        # Extract Metadata from API Response
        self.modality = api_response.get('fileType', 'Unknown').capitalize()
        metadata = api_response.get('metadata', {}) 
        duration = metadata.get('durationSeconds', 10)
        
        # Extract Prediction
        # data = str(metadata.get('prediction', '0'))
        raw_prediction = str(metadata.get('prediction', '0'))
        confidence = float(metadata.get('confidence', 0.0))
        
        # Calculate confidence based on the predicted label
        # prob_truth = data.get('truthful_prob', 0.0)
        # prob_decep = data.get('deceptive_prob', 0.0)

        if raw_prediction == '1':
            pred_label = 'DECEPTIVE'
        else:
            pred_label = 'TRUTHFUL'

        # UNCERTAIN label
        if confidence < 0.60:
            pred_label = 'uncertain'

        
        # Store processed chunk data
        self.chunks.append({
            'chunk_id': api_response.get('chunkId'),
            'duration': duration,
            'label': pred_label.upper(),
            'score': confidence
        })

    def generate(self):
        # Sort chunks by ID
        self.chunks.sort(key=lambda x: x['chunk_id'])

        # Calculate Aggregates
        total_seconds = sum(c['duration'] for c in self.chunks)
        total_chunks = len(self.chunks)
        
        # Format Duration
        mins, secs = divmod(int(total_seconds), 60)
        duration_str = f"{mins:02d}:{secs:02d} ({int(total_seconds)} seconds)"
        
        # Count stats
        deceptive_count = sum(1 for c in self.chunks if c['label'] == 'DECEPTIVE')
        if total_chunks > 0:
            avg_conf = sum(c['score'] for c in self.chunks) / total_chunks
            deception_ratio = deceptive_count / total_chunks
        else:
            avg_conf = 0
            deception_ratio = 0

        # Risk Level
        if deception_ratio >= 0.5:
            risk_level = "🔴 HIGH"
        elif deception_ratio >= 0.2:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"

        # Build Text Report
        lines = []
        curr_date = datetime.date.today().strftime("%B %d, %Y")
        
        # Header
        lines.append("Deception Analysis Report")
        lines.append(f"Case Reference ID: {self.case_id}")
        lines.append(f"Date: {curr_date}")
        lines.append(f"File Name: {self.file_name}")
        lines.append(f"Total Duration: {duration_str}")
        lines.append(f"Modality Analyzed: {self.modality}\n")
        
        # Executive Summary
        lines.append("1. Executive Summary")
        lines.append("A high-level overview of the entire file.")
        lines.append(f"Overall Risk Level: {risk_level}")
        lines.append(f"Average Confidence: {avg_conf:.0%}")
        lines.append(f"Flagged Segments: {deceptive_count} out of {total_chunks} chunks")
        lines.append(f"Primary Modality Trigger: {self.modality}")
        
        lines.append("\nKey")
        lines.append("Overall Risk Level: Determined by how often the “deception” label appears")
        lines.append("Average Confidence: Indicates the level of confidence in the final label.")
        lines.append("Flagged Segments: The total number of deception chunks in the input.")
        lines.append("Primary Modality Trigger: The primary input for determining the label.\n")
        
        # Detailed Segment Analysis
        lines.append("2. Detailed Segment Analysis (10s Chunks)")
        lines.append("Breakdown of the file into 10-second analysis windows.")
        header = f"{'Time Segment':<20} {'Label':<15} {'Confidence Score'}"
        lines.append(header)
        
        current_time = 0
        timeline_icons = []
        
        for chunk in self.chunks:
            start_s = current_time
            end_s = current_time + chunk['duration']
            
            time_str = f"{int(start_s//60):02d}:{int(start_s%60):02d} - {int(end_s//60):02d}:{int(end_s%60):02d}"
            label = chunk['label']
            conf = f"{chunk['score']:.0%}"
            
            lines.append(f"{time_str:<20} {label:<15} {conf}")
            
            # Map for visual
            if label == 'TRUTHFUL': timeline_icons.append((time_str.split(' - ')[0], "🟢"))
            elif label == 'DECEPTIVE': timeline_icons.append((time_str.split(' - ')[0], "🔴"))
            else: timeline_icons.append((time_str.split(' - ')[0], "🟡"))
            
            current_time += chunk['duration']

        lines.append("\n")
        
        # Timeline Visualization
        lines.append("3. Timeline Visualization")
        lines.append("A visual map of the 10-second chunks.")
        for time_mark, icon in timeline_icons:
            lines.append(f"{time_mark} {icon}")
            
        lines.append("\nKey:")
        lines.append("🟢 Truthful \n🟡 Uncertain \n🔴 Deceptive \n")
        
        # Disclaimer
        lines.append("4. Disclaimer")
        lines.append("This report is generated by an Artificial Intelligence system designed to detect statistical patterns associated with deception. These results are probabilistic and should be used as a support tool for human decision-making, not as an absolute determination of truth. Further manual review is recommended for all segments flagged as ”Deceptive”.")
        
        return "\n".join(lines)

# ==========================================
# TEST
# ==========================================

'''
{"sessionId": "test-session-002",
"fileType": "tensor",
"chunkId": "chunk_01", 
"s3Output": "s3://deception-detection-bucket/results/test-session-002/video/chunk_01/inference.json",
"status": "success", 
"metadata": {"originalResolution": {"width": 0, "height": 0}, 
                "numFrames": 64, 
                "prediction": "1", 
                "confidence": 0.8952397108078003}, 
"error": null}
'''


# CHUNK 1 Data (Deceptive)
response_01 = {
  "sessionId": "session_123",
  "fileType": "video",
  "chunkId": "chunk_01",
  "s3Output": "s3://deceptive-detection-bucket/results",
  "status": "success",
  "metadata": { "originalResolution": {"width": 1920, "height": 1080},
                "numFrames": 64, 
                "prediction": "0", 
                "confidence": 0.8952397108078003,
                "durationSeconds": 10 }
}

# json_01 = [
#     {"feature_path": "/tmp/tmpceq98k7c.pt",
#     "truthful_prob": 0.1047,
#     "deceptive_prob": 0.8952,
#     "predicted_label": "Deceptive"}
# ]

# CHUNK 2 Data (Truthful)
response_02 = {
  "sessionId": "session_123",
  "fileType": "video",
  "chunkId": "chunk_02",
  "s3Output": "s3://deceptive-detection-bucket/results",
  "status": "success",
  "metadata": { "originalResolution": {"width": 1920, "height": 1080},
                "numFrames": 64, 
                "prediction": "1", 
                "confidence": 0.8765,
                "durationSeconds": 10}
}

# json_02 = [
#     {"feature_path": "/tmp/tmpceq98677s.pt",
#     "truthful_prob": 0.9210,
#     "deceptive_prob": 0.0790,
#     "predicted_label": "Truthful"}
# ]

# CHUNK 3 Data (Truthful)
response_03 = {
  "sessionId": "session_123",
  "fileType": "video",
  "chunkId": "chunk_03",
  "s3Output": "s3://deceptive-detection-bucket/results",
  "status": "success",
  "metadata": { "originalResolution": {"width": 1920, "height": 1080},
                "numFrames": 64, 
                "prediction": "1", 
                "confidence": 0.536,
                "durationSeconds": 5}
}

# json_03 = [
#     {"feature_path": "/tmp/tmpceq9856fs.pt",
#     "truthful_prob": 0.85,
#     "deceptive_prob": 0.15,
#     "predicted_label": "Truthful"}
# ]


report_gen = DeceptionReportGenerator(case_id=response_01['sessionId'], file_name="interview.mp4")

report_gen.add_chunk(response_01)
report_gen.add_chunk(response_02)
report_gen.add_chunk(response_03)

print(report_gen.generate())