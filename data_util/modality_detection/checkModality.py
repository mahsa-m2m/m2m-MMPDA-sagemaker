import boto3
import filetype

# Initialize S3 client 
s3_client = boto3.client('s3')

def get_modality(file_header_bytes):
    """
    Determines if content is Video, Audio, or Text using pure python methods.
    """
    # Guess the file type using binary signatures
    kind = filetype.guess(file_header_bytes)

    if kind:
        mime = kind.mime
        print(f"Detected MIME: {mime}")
        
        if mime.startswith('video'):
            return 'video'
        if mime.startswith('audio'):
            return 'audio'
        if mime == 'application/pdf':
            return 'text'

    # If 'filetype' returns None, it might be a plain text file (.txt, .csv)
    #    Plain text files do not have magic headers, so we check if they are readable text.
    try:
        file_header_bytes.decode('utf-8')
        return 'text'
    except UnicodeDecodeError:
        pass

    return 'unknown'

def lambda_handler(event, context):
    try:
        # Parse S3 event
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
        
        print(f"Checking file: {file_key}")

        # Download only the first 2KB
        response = s3_client.get_object(
            Bucket=bucket_name, 
            Key=file_key, 
            Range='bytes=0-2047'
        )
        file_header = response['Body'].read()
        
        # Determine Modality
        modality = get_modality(file_header)
        
        print(f"FINAL DECISION: {modality}")
        
        return {
            'statusCode': 200,
            'body': modality
        }

    except Exception as e:
        print(f"Error: {e}")
        return {'statusCode': 500, 'body': str(e)}


if __name__ == "__main__":
    
    # Test
    with open("/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/sample/feat/train/trial_lie_033.pt", "rb") as f:
        print(f"Test.txt detected as: {get_modality(f.read(2048))}")
 