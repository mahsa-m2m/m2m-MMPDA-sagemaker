import os
from types import SimpleNamespace

# --- PATHS ---
# MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "model/best_model_loss_acc_72.pt")
MODEL_WEIGHTS_PATH = "/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/model/best_model_loss_ep3_acc65.pt"
DEVICE = "cuda" #if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"

# --- DATA PROCESSING CONFIG ---
NUM_FRAMES = 64 # T
FRAME_SIZE = (224, 224)
AUDIO_LENGTH = 80000 
SAMPLE_RATE = 16000
N_MELS = 128

# --- MODEL ARGUMENTS ---
MODEL_ARGS = SimpleNamespace(
    # Input dimensions
    a_dim = 512,  # Audio embedding dim (ResNet18_audio output)
    v_dim = 64,  # Vision/Behavior embedding dim (AU_GAZE... output)
    f_dim = 512,  # Face embedding dim (ResNet18_LSTM output)
    fusion_type = 'mult',
    modalities = 'vaf',
    common_dim = 128,
    num_heads = 8,
    mult_layer = 4,
    attn_dropout = 0.1,
    relu_dropout = 0.1,
    res_dropout = 0.1,
    embed_dropout = 0.0,
    attn_mask = False,
    
    # Any other args sub-models might need
    bidirectional = False 
)