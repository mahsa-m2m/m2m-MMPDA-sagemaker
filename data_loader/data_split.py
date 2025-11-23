import boto3
import random

BUCKET_NAME = 'deception-detection-bucket'
SOURCE_FOLDERS = ['dataset/video/deceptive', 'dataset/video/truthful']
DEST_BASE = 'dataset/video/splitted'
TRAIN_RATIO = 0.8

s3 = boto3.client('s3')

def list_objects(prefix):
    """List all objects under a given prefix."""
    objects = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Skip the folder itself
                if key != prefix and key != prefix + '/':
                    objects.append(key)
    return objects

def copy_object(source_key, dest_key):
    """Copy an object within the same bucket."""
    s3.copy_object(
        Bucket=BUCKET_NAME,
        CopySource={'Bucket': BUCKET_NAME, 'Key': source_key},
        Key=dest_key
    )
    print(f"Copied: {source_key} -> {dest_key}")

def split_and_copy(source_folder):
    """Split files from source folder into train/val destinations."""
    # Get the category name (deceptive or truthful)
    category = source_folder.rstrip('/').split('/')[-1]
    
    # List all files in source folder
    files = list_objects(source_folder)
    print(f"\nFound {len(files)} files in {source_folder}")
    
    # Shuffle for random split
    random.shuffle(files)
    
    # Calculate split index
    split_idx = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    print(f"Splitting into {len(train_files)} train, {len(val_files)} val")
    
    # Copy to train folder
    for src_key in train_files:
        filename = src_key.split('/')[-1]
        dest_key = f"{DEST_BASE}/train/{category}/{filename}"
        copy_object(src_key, dest_key)
    
    # Copy to val folder
    for src_key in val_files:
        filename = src_key.split('/')[-1]
        dest_key = f"{DEST_BASE}/val/{category}/{filename}"
        copy_object(src_key, dest_key)

def main():
    random.seed(42)  # For reproducibility
    
    for folder in SOURCE_FOLDERS:
        split_and_copy(folder)
    
    print("\nDataset split complete!")

if __name__ == '__main__':
    main()