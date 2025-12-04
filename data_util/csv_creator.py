# import boto3
# import re

# # ==========================================
# # CONFIGURATION
# # ==========================================
# BUCKET_NAME = 'deception-detection-bucket' 
# PREFIXES = {
#     'truthful': 'dataset/video/truthful/',
#     'deceptive': 'dataset/video/deceptive/'
# }

# # Pattern: Starts with W, then underscore, then 7
# PATTERN_REGEX = r"^W_7.*"

# # ⚠️ SAFETY SWITCH ⚠️
# # Set to True to SEE what will happen without deleting.
# # Set to False to ACTUALLY DELETE files.
# DRY_RUN = False 

# s3 = boto3.client('s3')

# def delete_files_with_pattern(bucket, prefixes):
#     total_deleted = 0
    
#     # FIX: Use .values() to get the actual paths ('dataset/video/...')
#     for prefix in prefixes.values():
#         paginator = s3.get_paginator('list_objects_v2')
#         pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
#         # We need to batch deletions (S3 accepts max 1000 per request)
#         batch_to_delete = []
        
#         print(f"Scanning folder: {prefix}")
        
#         for page in pages:
#             if 'Contents' not in page:
#                 continue
                
#             for obj in page['Contents']:
#                 full_key = obj['Key']
#                 filename = full_key.split('/')[-1]
                
#                 # Check regex match
#                 if re.match(PATTERN_REGEX, filename, re.IGNORECASE):
#                     # Add to batch
#                     batch_to_delete.append({'Key': full_key})
                    
#                     # If batch is full (1000 items), execute deletion
#                     if len(batch_to_delete) >= 1000:
#                         process_batch(bucket, batch_to_delete)
#                         total_deleted += len(batch_to_delete)
#                         batch_to_delete = [] # Reset batch

#         # Process any remaining files in the batch
#         if len(batch_to_delete) > 0:
#             process_batch(bucket, batch_to_delete)
#             total_deleted += len(batch_to_delete)

#     print("\n" + "="*40)
#     if DRY_RUN:
#         print(f"DRY RUN COMPLETE. {total_deleted} files WOULD have been deleted.")
#         print("Set DRY_RUN = False to execute actual deletion.")
#     else:
#         print(f"OPERATION COMPLETE. {total_deleted} files were permanently deleted.")
#     print("="*40)

# def process_batch(bucket, objects_list):
#     """
#     Helper function to delete a list of objects or print them if Dry Run
#     """
#     if not objects_list:
#         return

#     if DRY_RUN:
#         for obj in objects_list:
#             print(f"[DRY RUN] Would delete: {obj['Key']}")
#     else:
#         # Perform actual deletion
#         response = s3.delete_objects(
#             Bucket=bucket,
#             Delete={
#                 'Objects': objects_list,
#                 'Quiet': True
#             }
#         )
#         # Check for errors
#         if 'Errors' in response:
#             print(f"Errors occurred: {response['Errors']}")
#         else:
#             print(f"Deleted batch of {len(objects_list)} files...")

# # ==========================================
# # EXECUTION
# # ==========================================

# print(f"Starting Process... (DRY_RUN is {DRY_RUN})")
# delete_files_with_pattern(BUCKET_NAME, PREFIXES)



## ========================================================================================================================================================================
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURATION ---
BUCKET_NAME = 'deception-detection-bucket'  
PREFIX_TRUTHFUL = 'dataset/video/truthful/'
PREFIX_DECEPTION = 'dataset/video/deceptive/'

# Where to save the CSVs inside the bucket
OUTPUT_PREFIX = 'dataset/video/'

s3 = boto3.client('s3')

# ==========================================
# HELPER: Get all files from S3 folder
# ==========================================
def get_s3_files(bucket, prefix, label):
    """
    Lists all files in an S3 folder (handles pagination for large datasets)
    and returns a list of dictionaries containing the S3 path and label.
    """
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    data = []
    
    print(f"Scanning {prefix}...")
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Skip the folder object itself (if it exists)
                if key.endswith('/'):
                    continue
                    
                # Construct full S3 path
                s3_path = f"s3://{bucket}/{key}"
                data.append({'s3_path': s3_path, 'label': label})
                
    print(f"Found {len(data)} files in {prefix}")
    return data

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Fetch data lists
truthful_data = get_s3_files(BUCKET_NAME, PREFIX_TRUTHFUL, 'truthful')
deception_data = get_s3_files(BUCKET_NAME, PREFIX_DECEPTION, 'deceptive')

# 2. Create a single Master DataFrame
full_data = truthful_data + deception_data
df = pd.DataFrame(full_data)

# Shuffle the data initially to ensure randomness before splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total dataset size: {len(df)}")

# 3. Split the data
# We need 70% Train, 15% Val, 15% Test.
# First, split into Train (70%) and Temp (30%)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.30, 
    stratify=df['label'], # Ensures distribution is similar in split
    random_state=42
)

# Next, split Temp (30%) into Validation (15%) and Test (15%)
# Since 15% is exactly half of 30%, we split Temp by 0.5
val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50, 
    stratify=temp_df['label'], 
    random_state=42
)

# 4. Balance the Training Set
# You requested the Train set be balanced for both classes.
# We will undersample the majority class in the training set.

# Separate the classes
train_truthful = train_df[train_df['label'] == 'truthful']
train_deception = train_df[train_df['label'] == 'deceptive']

# Find which class is smaller
min_count = min(len(train_truthful), len(train_deception))

# Sample both to the minimum count
balanced_truthful = train_truthful.sample(n=min_count, random_state=42)
balanced_deception = train_deception.sample(n=min_count, random_state=42)

# Recombine and shuffle
train_df_balanced = pd.concat([balanced_truthful, balanced_deception])
train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("-" * 30)
print("SPLIT SUMMARY")
print("-" * 30)
print(f"Training Set (Balanced): {len(train_df_balanced)} rows")
print(train_df_balanced['label'].value_counts())
print("-" * 30)
print(f"Validation Set: {len(val_df)} rows")
print(val_df['label'].value_counts())
print("-" * 30)
print(f"Test Set: {len(test_df)} rows")
print(test_df['label'].value_counts())

# 5. Save locally and Upload to S3
def save_and_upload(dataframe, filename):
    # Save locally
    local_path = filename
    dataframe.to_csv(local_path, index=False, header=False) # No header usually preferred for training scripts, set True if needed
    
    # Upload to S3
    s3_key = f"{OUTPUT_PREFIX}{filename}"
    s3_dest = f"s3://{BUCKET_NAME}/{s3_key}"
    
    dataframe.to_csv(s3_dest, index=False)
    print(f"Saved {filename} to {s3_dest}")

save_and_upload(train_df_balanced, 'train.csv')
save_and_upload(val_df, 'validation.csv')
save_and_upload(test_df, 'test.csv')

print("\nProcessing Complete!")

# ======================== LOCAL ============================
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os

# # --- CONFIGURATION ---
# # Define the path to your local folders
# # '.' represents the current directory where this script runs
# BASE_DIR = '/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/sample/train' 
# PATH_TRUTHFUL = os.path.join(BASE_DIR, 'truthful')
# PATH_DECEPTION = os.path.join(BASE_DIR, 'deceptive')

# # Where to save the resulting CSVs
# OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# # ==========================================
# # HELPER: Get all files from local folder
# # ==========================================
# def get_local_files(directory, label):
#     """
#     Lists all files in a local directory recursively
#     and returns a list of dictionaries containing the file path and label.
#     """
#     data = []
    
#     # Check if directory exists to avoid errors
#     if not os.path.exists(directory):
#         print(f"WARNING: Directory not found: {directory}")
#         return data

#     print(f"Scanning {directory}...")
    
#     # os.walk allows recursive searching (subfolders)
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             # Optional: Filter for specific file types if needed (e.g., .mp4, .avi)
#             # if not file.endswith(('.mp4', '.avi', '.mov')):
#             #     continue

#             # Skip hidden files (like .DS_Store on Mac)
#             if file.startswith('.'):
#                 continue
                
#             # Construct absolute local path
#             file_path = os.path.abspath(os.path.join(root, file))
#             data.append({'file_path': file_path, 'label': label})
                
#     print(f"Found {len(data)} files in {directory}")
#     return data

# # ==========================================
# # MAIN EXECUTION
# # ==========================================

# # 1. Fetch data lists
# truthful_data = get_local_files(PATH_TRUTHFUL, 'truthful')
# deception_data = get_local_files(PATH_DECEPTION, 'deceptive')

# # 2. Create a single Master DataFrame
# full_data = truthful_data + deception_data

# if not full_data:
#     print("Error: No files found. Please check your folder paths.")
#     exit()

# df = pd.DataFrame(full_data)

# # Shuffle the data initially to ensure randomness before splitting
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# print(f"Total dataset size: {len(df)}")

# # 3. Split the data
# # We need 70% Train, 15% Val, 15% Test.
# # First, split into Train (70%) and Temp (30%)
# train_df, temp_df = train_test_split(
#     df, 
#     test_size=0.30, 
#     stratify=df['label'], # Ensures distribution is similar in split
#     random_state=42
# )

# # Next, split Temp (30%) into Validation (15%) and Test (15%)
# val_df, test_df = train_test_split(
#     temp_df, 
#     test_size=0.50, 
#     stratify=temp_df['label'], 
#     random_state=42
# )

# # 4. Balance the Training Set
# # We will undersample the majority class in the training set.

# # Separate the classes
# train_truthful = train_df[train_df['label'] == 'truthful']
# # NOTE: Corrected 'Deception' to 'deceptive' to match the input label defined in Step 1
# train_deception = train_df[train_df['label'] == 'deceptive'] 

# # Check if we have data for both classes
# if len(train_truthful) == 0 or len(train_deception) == 0:
#     print("Error: One of the classes in the training set is empty. Cannot balance.")
#     print(f"Truthful count: {len(train_truthful)}, Deceptive count: {len(train_deception)}")
#     exit()

# # Find which class is smaller
# min_count = min(len(train_truthful), len(train_deception))

# # Sample both to the minimum count
# balanced_truthful = train_truthful.sample(n=min_count, random_state=42)
# balanced_deception = train_deception.sample(n=min_count, random_state=42)

# # Recombine and shuffle
# train_df_balanced = pd.concat([balanced_truthful, balanced_deception])
# train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# print("-" * 30)
# print("SPLIT SUMMARY")
# print("-" * 30)
# print(f"Training Set (Balanced): {len(train_df_balanced)} rows")
# print(train_df_balanced['label'].value_counts())
# print("-" * 30)
# print(f"Validation Set: {len(val_df)} rows")
# print(val_df['label'].value_counts())
# print("-" * 30)
# print(f"Test Set: {len(test_df)} rows")
# print(test_df['label'].value_counts())

# # 5. Save locally
# def save_locally(dataframe, filename):
#     # Ensure output directory exists
#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)
        
#     local_path = os.path.join(OUTPUT_DIR, filename)
    
#     # Save CSV
#     # header=False is usually preferred for training scripts; set True if you want column names
#     dataframe.to_csv(local_path, index=False, header=False) 
    
#     print(f"Saved {filename} to {local_path}")

# save_locally(train_df_balanced, 'train.csv')
# save_locally(val_df, 'validation.csv')
# save_locally(test_df, 'test.csv')

# print("\nProcessing Complete!")