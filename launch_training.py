import sagemaker
from sagemaker.pytorch import PyTorch 
from sagemaker import get_execution_role
# from sagemaker.debugger import ProfilerConfig, ProfilerRule, rule_configs

# Get SageMaker session and role
sess = sagemaker.Session()
role = get_execution_role()

# Your custom bucket
BUCKET_NAME = 'deception-detection-bucket'

print(f"✓ Using role: {role}")
print(f"✓ Using bucket: {BUCKET_NAME}")

# CONFIGURE THE PROFILER
# This captures system metrics (GPU, CPU, RAM, I/O) every 500ms (0.5s)
# profiler_config = ProfilerConfig(
#     system_monitor_interval_millis=500,
#     s3_output_path=f's3://{BUCKET_NAME}/profiler-output/'
# )

# # Profiler Rules
# profiler_rules = [
#     ProfilerRule.sagemaker(rule_configs.ProfilerReport()),    # Generates the PDF report
#     ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()), # Alerts if GPU is sitting idle
#     ProfilerRule.sagemaker(rule_configs.CPUBottleneck())      
# ]

metric_definitions = [
    {'Name': 'epoch',      'Regex': 'Epoch ([0-9]+)/'},
    {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
    {'Name': 'train:acc',  'Regex': 'Train Acc: ([0-9\\.]+)%'},
    {'Name': 'val:loss',   'Regex': 'Val Loss: ([0-9\\.]+)'},    
    {'Name': 'val:acc',    'Regex': 'Val Acc: ([0-9\\.]+)%'},
]

# Create PyTorch estimator
estimator = PyTorch(
    entry_point='train_test_feature.py',
    source_dir='.',  # Uploads all files in current directory
    role=role,
    instance_type='ml.g5.12xlarge',  # ml.g4dn.xlarge = 1x NVIDIA T4 GPU, 16GB VRAM / ml.g5.xlarge / ml.g4dn.2xlarge / ml.g5.2xlarge / ml.g5.12xlarge
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',  # ← Changed to py310 (py311 not available)
    output_path=f's3://{BUCKET_NAME}/output/',
    metric_definitions=metric_definitions,

    environment={
        'PYTHONUNBUFFERED': '1'        # Forces prints to show up immediately
        , 'SM_CHECKPOINT_DIR': '/opt/ml/checkpoints',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'

    },
    # CLI args
    hyperparameters={
        'batchsize': 16,
        'max_epochs': 30,
        # 'fusion_type': 'mult',
        'num_frames': 64,
        'lr': 1e-5,
        'num_workers': 40,
        'train_root': '/opt/ml/input/data/training',
        'val_root':   '/opt/ml/input/data/validation',
        'audio_length': 80000,
        'frame_height': 224,
        'frame_width': 224
    },
    
    # Spot instances (cheaper but can be interrupted)
    use_spot_instances=True,
    max_run=86400,      # 24 hours max training time
    max_wait=90000,     # 13 hours max wait (including spot delays)
    
    volume_size=100,    # GB of EBS storage
    
    # Checkpointing - saves to S3, restores on spot interruption
    checkpoint_s3_uri=f's3://{BUCKET_NAME}/checkpoints/',
    checkpoint_local_path='/opt/ml/checkpoints',
    
    # Dependencies
    dependencies=['requirements.txt'],
    
#     # PROFILER CONFIG
#     profiler_config=profiler_config,
#     rules=profiler_rules
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

# sess.logs_for_job(job_name=estimator.latest_training_job.name, wait=True)
