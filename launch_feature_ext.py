import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

sess = sagemaker.Session()
role = get_execution_role()

BUCKET_NAME = 'deception-detection-bucket'

# 1. Input: Video & CSVs
s3_data_root = f's3://{BUCKET_NAME}/dataset/video/'

# 2. Output: Where features will be saved
# We use this as the "Checkpoint" path so files are uploaded immediately, not zipped
s3_output_root = f's3://{BUCKET_NAME}/dataset/video/precomputed_features/'

print(f"🚀 Launching Spot Feature Extraction...")
print(f"   Input:  {s3_data_root}")
print(f"   Output: {s3_output_root}")

# 3. Configure Estimator (Training Job)
estimator = PyTorch(
    entry_point='data_util/feature_extractor.py', # Your existing script
    source_dir='.',
    role=role,
    instance_type='ml.c5.18xlarge',
    instance_count=1,
    framework_version='2.0',
    py_version='py310',
    base_job_name='mmpda-spot-extract',
    
    # --- SPOT INSTANCE CONFIGURATION ---
    use_spot_instances=True,
    max_run=86400,          # Max run time (24 hours)
    max_wait=86400,         # Max wait time (must be >= max_run)
    
    # Disk Size: 179GB Input + 460GB Output + Buffer
    volume_size=1024,
    
    # --- OUTPUT MAGIC ---
    # We map the local checkpoint folder to your S3 Output bucket.
    # SageMaker syncs this folder to S3 continuously.
    checkpoint_s3_uri=s3_output_root,
    checkpoint_local_path='/opt/ml/checkpoints',
    
    # Pass arguments to your script
    hyperparameters={
        # Input mapped by SageMaker to 'training' channel
        'data_dir': '/opt/ml/input/data/training',
        'csv_dir':  '/opt/ml/input/data/training',
        
        # Output mapped to the checkpoint folder so it syncs to S3
        'output_dir': '/opt/ml/checkpoints',
        
        'train_csv': 'train.csv', 
        'val_csv': 'validation.csv',
        'test_csv': 'test.csv'
    }
)

# 4. Launch
estimator.fit(
    inputs={
        # This downloads all files from s3_data_root to /opt/ml/input/data/training
        'training': s3_data_root
    },
    wait=False
)

print(f"\n✅ Spot Job submitted successfully!")
print(f"Job Name: {estimator.latest_training_job.job_name}")
print(f"Check status: https://console.aws.amazon.com/sagemaker/home#/jobs/{estimator.latest_training_job.job_name}")
# import sagemaker
# from sagemaker.pytorch.processing import PyTorchProcessor
# from sagemaker.processing import ProcessingInput, ProcessingOutput
# from sagemaker import get_execution_role

# sess = sagemaker.Session()
# role = get_execution_role()

# BUCKET_NAME = 'deception-detection-bucket'


# # 1. Input Path
# # This root contains 'train.csv', 'validation.csv', 'test.csv', 'truthful/', 'deceptive/'
# s3_data_root = f's3://{BUCKET_NAME}/dataset/video/'

# # 2. Output Path
# # This is where the .pt files will land (inside 'train/', 'val/', 'test/' subfolders)
# s3_output_root = f's3://{BUCKET_NAME}/dataset/video/precomputed_features/'

# print(f"🚀 Launching extraction job...")
# print(f"   Input:  {s3_data_root}")
# print(f"   Output: {s3_output_root}")


# # 3. Configure Processor
# # Using ml.c5.4xlarge (CPU optimized)
# processor = PyTorchProcessor(
#     framework_version='2.0',
#     role=role,
#     instance_type='ml.c5.18xlarge', #ml.c5.4xlarge
#     instance_count=1,
#     base_job_name='mmpda-feat-extract',
#     py_version='py310',
#     volume_size_in_gb=1024
# )

# processor.run(
#     code='data_util/feature_extractor.py',
#     source_dir='.', # Uploads current directory content
    
#     inputs=[
#         # Mount the entire video/csv directory to /opt/ml/processing/input/data
#         ProcessingInput(
#             source=s3_data_root,
#             destination='/opt/ml/processing/input/data',
#             input_name='dataset_root',
#             s3_input_mode='File'
#         )
#     ],
    
#     outputs=[
#         # Save results back to S3
#         ProcessingOutput(
#             source='/opt/ml/processing/output',
#             destination=s3_output_root,
#             output_name='features'
#         )
#     ],
    
#        arguments=[
#         # Since 's3_data_root' contains both videos AND csvs, they are both in /data
#         '--csv_dir', '/opt/ml/processing/input/data',
#         '--data_dir', '/opt/ml/processing/input/data',
        
#         '--train_csv', 'train.csv', 
#         '--val_csv', 'validation.csv',
#         '--test_csv', 'test.csv'
#     ],
    
#     wait=False 
# )

# print(f"\n✅ Job submitted successfully!")
# print(f"Job Name: {processor.latest_job.job_name}")
# print(f"Check status here: https://console.aws.amazon.com/sagemaker/home#/processing-jobs/{processor.latest_job.job_name}")