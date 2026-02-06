import json
from datetime import datetime, timezone
import os             
import logging        
import boto3
from botocore.exceptions import ClientError 

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
ddb_client = boto3.client("dynamodb")

TABLE_NAME = os.environ.get("TABLE_NAME", "UploadSessions")
REPORT_BUCKET = os.environ.get("REPORT_BUCKET", "deception-detection-reports")

class DeceptionReportGenerator:
    def __init__(self, case_id, file_name):
        self.case_id = case_id
        self.file_name = file_name
        self.chunks = []
        self.modality = "Text"
        self.total_tokens = 0  
        self.chunk_size = 400  

    def add_chunk(self, api_response):
        """
        Parses the input.
        """

        try:
            # Extract Metadata from API Response
            # self.modality = api_response.get('fileType', 'Text').capitalize()
            # metadata = api_response.get('metadata', {})
            chunk_modality = api_response.get('primaryModality', 'Text').capitalize()
            
            # token_count = metadata.get('tokenCount', self.chunk_size)
            token_count = 400

            # Extract Prediction
            # raw_prediction = str(metadata.get('prediction', '0'))
            # confidence = float(metadata.get('confidence', 0.0))
            raw_prediction = str(api_response.get('prediction', '0'))
            confidence = float(api_response.get('confidence', 0.0))
            
            # Logic for Label Mapping (1=Deceptive, 0=Truthful)
            if raw_prediction == '1':
                pred_label = 'DECEPTIVE'
            else:
                pred_label = 'TRUTHFUL'

            # UNCERTAIN label logic
            if confidence < 0.60:
                pred_label = 'UNCERTAIN'

            # Calculate token range for the chunk
            start_token = self.total_tokens
            end_token = self.total_tokens + token_count
            self.total_tokens += token_count # Increment total

            # Store processed chunk data
            self.chunks.append({
                # Using numerical index
                'chunk_id': len(self.chunks) + 1, 
                'start_token': start_token,
                'end_token': end_token,
                'label': pred_label.upper(),
                'score': confidence
            })
        except Exception as e:
            logger.error(f"Error parsing chunk {api_response.get('chunkId', 'unknown')}: {str(e)}")

    def generate(self):
        # Sort chunks
        self.chunks.sort(key=lambda x: x['chunk_id'])

        total_chunks = len(self.chunks)
        
        # Count stats
        deceptive_count = sum(1 for c in self.chunks if c['label'] == 'DECEPTIVE')
        if total_chunks > 0:
            avg_conf = sum(c['score'] for c in self.chunks) / total_chunks
            deception_ratio = deceptive_count / total_chunks
        else:
            avg_conf = 0
            deception_ratio = 0

        # Risk Level Logic
        if deception_ratio >= 0.5:
            risk_level = "🔴 HIGH"
        elif deception_ratio >= 0.2:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"

        # Build Text Report
        lines = []
        # curr_date = datetime.date.today().strftime("%B %d, %Y")
        curr_date = datetime.now(timezone.utc).strftime("%B %d, %Y")

        
        # HEADER
        lines.append("Deception Analysis Report")
        lines.append(f"Case Reference ID: {self.case_id}")
        lines.append(f"Date: {curr_date}")
        lines.append(f"File Name: {self.file_name}")
        # Total Tokens and estimated chunks
        lines.append(f"Total Tokens: {self.total_tokens} (~ {total_chunks} chunks)")
        lines.append(f"Modality Analyzed: [{self.modality}]\n")
        
        # Executive Summary
        lines.append("1. Executive Summary")
        lines.append("A high-level overview of the entire file.")
        lines.append(f"Overall Risk Level: {risk_level}")
        lines.append(f"Average Confidence: {avg_conf:.0%}")
        lines.append(f"Flagged Segments: {deceptive_count} out of {total_chunks} chunks")
        
        # Key Definition 
        lines.append("\nKey")
        lines.append("Definition")
        lines.append("Overall Risk Level")
        lines.append("It is determined by how often the “deception” label appears in the input")
        lines.append("A higher value  = A higher risk = More deceptive")
        lines.append("Average Confidence")
        lines.append("It indicates the level of confidence in the final label.")
        lines.append("Flagged Segments")
        lines.append("The total number of deception chunks in the input.\n")
        
        # Detailed Segment Analysis
        lines.append("________________________________________________________________________________")
        lines.append("2. Detailed Segment Analysis") 
        lines.append(f"Breakdown of the file into {self.chunk_size}-token analysis windows.\n")
        
        header = f"{'Chunk ID: Tokens':<25} {'Label':<20} {'Confidence Score'}"
        lines.append(header)
        lines.append("") 

        timeline_icons_list = []
        
        for chunk in self.chunks:
            id_range = f"{chunk['chunk_id']}: {chunk['start_token']}-{chunk['end_token']}"
            label = chunk['label']
            conf = f"{chunk['score']:.0%}"
            
            lines.append(f"{id_range:<25} {label:<20} {conf}")
            
            # Map for visual
            if label == 'TRUTHFUL': timeline_icons_list.append("🟢")
            elif label == 'DECEPTIVE': timeline_icons_list.append("🔴")
            else: timeline_icons_list.append("🟡")

        lines.append("________________________________________________________________________________")
        lines.append("\n")
        
        # Segments Visualization
        lines.append("3. Segments Visualization")
        lines.append(f"A visual map of the {self.chunk_size}-token chunks.")
        lines.append("") 

        header_row = ""
        icon_row = ""
        col_width = 12
        
        for i, chunk in enumerate(self.chunks):
            # Append Header
            c_str = f"Chunk: {chunk['chunk_id']}"
            header_row += f"{c_str:<{col_width}}"
            
            # Append Icon
            icon = timeline_icons_list[i]
            icon_row += f"{icon:<{col_width}}"

        # Add the horizontal rows to the report
        lines.append(header_row.rstrip())
        lines.append("\n")
        lines.append(icon_row.rstrip())
            
        lines.append("\nKey:")
        lines.append("🟢 Truthful \n🟡 Uncertain \n🔴 Deceptive \n")
        
        # Pattern Logic
        # pattern_str = "Pattern: scattered throughout document"
        # if timeline_icons_list and timeline_icons_list[-1] == "🔴":
        #     pattern_str = "Pattern: Deception increases toward end of document"
            
        # lines.append(f"{pattern_str} \n")


        # Disclaimer
        lines.append("4. Disclaimer") 
        lines.append("This report is generated by an Artificial Intelligence system designed to detect statistical patterns associated with deception. These results are probabilistic and should be used as a support tool for human decision-making, not as an absolute determination of truth. Further manual review is recommended for all segments flagged as ”Deceptive”.")
        
        return "\n".join(lines)


def save_to_s3(session_id, report_content):
    key = f"reports/{session_id}/analysis.txt"
    s3_client.put_object(Bucket=REPORT_BUCKET, Key=key, Body=report_content)
    return key


def lambda_handler(event, context):
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        session_id = event.get('session_id')
        file_name = event.get('file_name', 'unknown')
        
        # a list of chunks to be passed in the event
        chunk_results = event.get('chunk_results', [])
        
        if not session_id:
            raise ValueError("session_id is missing from event")

        report_gen = DeceptionReportGenerator(case_id=session_id, file_name=file_name)

        # Process chunks
        for chunk in chunk_results:
            report_gen.add_chunk(chunk)

        # Generate string
        report_text = report_gen.generate()
        
        # save to S3
        saved_key = save_to_s3(session_id, report_text)
        logger.info(f"Report saved to {saved_key}")

        # Update DynamoDB
        ddb_client.update_item(
            TableName=TABLE_NAME,
            Key={"session_id": {"S": session_id}},
            UpdateExpression="SET #status = :s, report_key = :rk",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":s": {"S": "COMPLETED"},
                ":rk": {"S": saved_key}
            }
        )

        # JSON response
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Report generated",
                "report_s3_key": saved_key
            })
        }

    except Exception as e:
        logger.error(f"Lambda failed: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

# # # ==========================================
# # # TEST
# # # ==========================================

# text_response_01 = {
#       "primaryModality": "text",
#       "status": "success",
#       "confidence": 0.9134287238121033,
#       "prediction": "1",
#       "sessionId": "session_d5396f09ee13_1770332044",
#       "chunkId": "chunk_1"
#     }
# # Text Response 1 (Truthful)
# # text_response_01 = {
# #   "sessionId": "session_text_001",
# #   "fileType": "text",
# #   "s3Output": "s3://results/session/text/text_inference.json",
# #   "status": "success",
# #   "metadata": {"confidence": 0.92, "prediction": 0, "tokenCount": 350}, # 0 = Truthful
# #   "error": None
# # }

# # # Text Response 2 (Truthful)
# # text_response_02 = {
# #   "sessionId": "session_text_001",
# #   "fileType": "text",
# #    "s3Output": "s3://results/session/text/text_inference.json",
# #   "status": "success",
# #   "metadata": {"confidence": 0.89, "prediction": 0, "tokenCount": 350}
# # }

# # # Text Response 3 (Truthful)
# # text_response_03 = {
# #   "sessionId": "session_text_001",
# #   "fileType": "text",
# #   "s3Output": "s3://results/session/text/text_inference.json",
# #   "status": "success",
# #   "metadata": {"confidence": 0.85, "prediction": 0, "tokenCount": 350}
# # }

# # # Text Response 4 (Uncertain)
# # text_response_04 = {
# #   "sessionId": "session_text_001",
# #   "fileType": "text",
# #   "status": "success",
# #   "metadata": {"confidence": 0.55, "prediction": 1, "tokenCount": 230} 
# # }

# # # Text Response 5 (Deceptive)
# # text_response_05 = {
# #   "sessionId": "session_text_001",
# #   "fileType": "text",
# #   "status": "success",
# #   "metadata": {"confidence": 0.88, "prediction": 1} 
# # }

# # # Text Response 6 (Deceptive)
# # text_response_06 = {
# #   "sessionId": "session_text_001",
# #   "fileType": "text",
# #   "status": "success",
# #   "metadata": {"confidence": 0.91, "prediction": 1}
# # }


# report_gen = DeceptionReportGenerator(case_id=text_response_01.get("sessionId"), file_name=".docx")

# report_gen.add_chunk(text_response_01) # Chunk 1
# report_gen.add_chunk(text_response_01) # Chunk 1
# report_gen.add_chunk(text_response_01) # Chunk 1
# # report_gen.add_chunk(text_response_02) # Chunk 2
# # report_gen.add_chunk(text_response_03) # Chunk 3
# # report_gen.add_chunk(text_response_04) # Chunk 4
# # report_gen.add_chunk(text_response_05) # Chunk 5
# # report_gen.add_chunk(text_response_06) # Chunk 6

# print(report_gen.generate())