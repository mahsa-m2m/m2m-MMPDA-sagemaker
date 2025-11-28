# Use SageMaker PyTorch base image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py311-cu121-ubuntu20.04-sagemaker

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies (matching your requirements comments)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install exact versions from your requirements.txt
# Installing in groups to optimize layer caching

# PyTorch ecosystem (should already be in base, but ensuring exact versions)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchaudio==2.2.2 \
    torchvision==0.17.2

# Core scientific computing
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.3.3 \
    scikit-learn==1.2.0

# Computer vision libraries
RUN pip install --no-cache-dir \
    opencv-python-headless==4.7.0.72 \
    opencv-contrib-python==4.11.0.86 \
    mediapipe==0.10.5 \
    Pillow==12.0.0 \
    imgaug==0.4.0

# Audio processing
RUN pip install --no-cache-dir \
    librosa==0.11.0

# Visualization
RUN pip install --no-cache-dir \
    matplotlib==3.10.7 \
    seaborn==0.13.2

# Utilities
RUN pip install --no-cache-dir \
    absl-py==2.1.0 \
    thop==0.1.1.post2209072238 \
    tqdm==4.66.2

# Verify critical installations
RUN python -c "import torch; print(f'PyTorch {torch.__version__} (CUDA available: {torch.cuda.is_available()})')" && \
    python -c "import cv2; print(f'OpenCV {cv2.__version__}')" && \
    python -c "import librosa; print(f'Librosa {librosa.__version__}')" && \
    python -c "import mediapipe; print(f'MediaPipe {mediapipe.__version__}')" && \
    ffmpeg -version | head -n 1

# Set the training script as the entrypoint
ENV SAGEMAKER_PROGRAM=train_test_feature.py

# Print success message during build
RUN echo "✅ Docker image built successfully with all dependencies!"