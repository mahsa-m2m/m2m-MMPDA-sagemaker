# Use SageMaker PyTorch base image with Python 3.11
# For GPU: use pytorch-training:2.2.0-gpu-py311
# For CPU: use pytorch-training:2.2.0-cpu-py311
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py311-cu121-ubuntu20.04-sagemaker

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch and related (matching your environment)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchaudio==2.2.2 \
    torchvision==0.17.2 \
    triton==2.2.0

# Install computer vision and audio processing libraries
RUN pip install --no-cache-dir \
    opencv-python-headless==4.7.0.72 \
    opencv-contrib-python==4.11.0.86 \
    mediapipe==0.10.21 \
    librosa==0.11.0 \
    soundfile==0.13.1 \
    sounddevice==0.5.3 \
    audioread==3.1.0 \
    soxr==1.0.0

# Install scientific computing libraries
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.16.3 \
    pandas==2.3.3 \
    scikit-learn==1.2.0 \
    scikit-image==0.25.2

# Install visualization libraries
RUN pip install --no-cache-dir \
    matplotlib==3.10.7 \
    seaborn==0.13.2 \
    Pillow==12.0.0 \
    imageio==2.37.2 \
    tifffile==2025.10.16

# Install deep learning utilities
RUN pip install --no-cache-dir \
    imgaug==0.4.0 \
    thop==0.1.1.post2209072238

# Install SageMaker dependencies
RUN pip install --no-cache-dir \
    sagemaker==3.0.1 \
    sagemaker-training==4.10.2 \
    boto3==1.35.92 \
    botocore==1.35.92

# Install other required packages
RUN pip install --no-cache-dir \
    protobuf==4.25.8 \
    click==8.3.1 \
    tqdm==4.66.2 \
    omegaconf==2.3.0 \
    PyYAML==6.0.3

# Copy all your code
COPY train_test_feature.py .
COPY DALoss.py .
COPY DANetwork.py .
COPY utils.py .
COPY data_loader/ ./data_loader/
COPY models/ ./models/
COPY models_comp/ ./models_comp/
COPY modules/ ./modules/

# Verify FFmpeg installation
RUN ffmpeg -version

# Set the training script as the entrypoint
ENV SAGEMAKER_PROGRAM=train_test_feature.py

# Default command (will be overridden by SageMaker)
CMD ["python", "train_test_feature.py"]