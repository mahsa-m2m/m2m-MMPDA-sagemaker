import sagemaker
from sagemaker.pytorch import PyTorch 
from sagemaker import get_execution_role

# Get SageMaker session and role
sess = sagemaker.Session()
role = get_execution_role()

# Your custom bucket
BUCKET_NAME = 'deception-detection-bucket'

print(f"✓ Using role: {role}")
print(f"✓ Using bucket: {BUCKET_NAME}")

# Create PyTorch estimator
estimator = PyTorch(
    entry_point='train_test_feature.py',
    source_dir='.',  # Uploads all files in current directory
    role=role,
    instance_type='g4dn.2xlarge',  # ml.g4dn.xlarge = 1x NVIDIA T4 GPU, 16GB VRAM / ml.g5.xlarge / g4dn.2xlarge
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',  # ← Changed to py310 (py311 not available)
    output_path=f's3://{BUCKET_NAME}/output/',
    
    environment={
        'PYTHONUNBUFFERED': '0',        # Forces prints to show up immediately
        'SM_CHECKPOINT_DIR': '/opt/ml/checkpoints' 
    },
    # These become CLI args: --train_root, --val_root, etc.
    hyperparameters={
        'batchsize': 16,
        'max_epochs': 30,
        'fusion_type': 'mult',
        'num_frames': 32,
        'lr': 5e-3,
        'train_root': '/opt/ml/input/data/training',
        'val_root':   '/opt/ml/input/data/validation',
    },
    
    # Spot instances (cheaper but can be interrupted)
    use_spot_instances=True,
    max_run=43200,      # 12 hours max training time
    max_wait=46800,     # 13 hours max wait (including spot delays)
    
    volume_size=100,    # GB of EBS storage
    
    # Checkpointing - saves to S3, restores on spot interruption
    checkpoint_s3_uri=f's3://{BUCKET_NAME}/checkpoints/',
    checkpoint_local_path='/opt/ml/checkpoints',
    
    # Dependencies
    dependencies=['requirements.txt'],
)

# Map channel names to S3 paths
# SageMaker downloads these to /opt/ml/input/data/{channel_name}/
print("\n🚀 Submitting training job...")
estimator.fit({
    'training': f's3://{BUCKET_NAME}/dataset/video/splitted/train/',  # ← Note: 'training' not 'train'
    'validation': f's3://{BUCKET_NAME}/dataset/video/splitted/val/'   # ← Note: 'validation' not 'val'
}, wait=True)

print(f"\n✅ Training job submitted!")
print(f"Job name: {estimator.latest_training_job.name}")
print(f"📊 Monitor at: https://console.aws.amazon.com/sagemaker/home#/jobs/{estimator.latest_training_job.name}")