import pandas as pd
import os

# 1. Setup file names (Change these to match your actual filenames)
input_csv = '/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/tests/test.csv'
output_csv = 'data_updated.csv'

# 2. Define the new target directory prefix
# This is the part before the filename in your desired output
new_folder_path = "s3://deception-detection-bucket/dataset/video/precomputed_features/test/"

def transform_s3_path(old_path):
    """
    Takes an old S3 path, extracts the filename, removes the extension,
    and returns the new path with .pt extension.
    """
    # Extract the filename from the full path (e.g., TTTT_432...mkv)
    filename = os.path.basename(old_path)
    
    # Split the filename into name and extension (e.g., TTTT_432..., .mkv)
    file_stem, _ = os.path.splitext(filename)
    
    # Construct the new string: New Folder + Filename Stem + .pt
    return f"{new_folder_path}{file_stem}.pt"

# 3. Read the CSV
try:
    df = pd.read_csv(input_csv)
    
    # 4. Apply the transformation to the 's3_path' column
    df['s3_path'] = df['s3_path'].apply(transform_s3_path)
    
    # 5. Save the updated CSV
    df.to_csv(output_csv, index=False)
    
    print(f"Success! Processed {len(df)} rows.")
    print(f"Saved to: {output_csv}")
    
    # Optional: Print the first row to verify
    print("\nPreview of updated data:")
    print(df.head(1))

except FileNotFoundError:
    print(f"Error: Could not find file '{input_csv}'")
