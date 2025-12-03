import torch

# Load the file you saved
checkpoint = torch.load('/home/sagemaker-user/mahsa-m2m-MMPDA-sagemaker/checkpoints/best_model_epoch_1.pt')
keys = checkpoint.keys() # or checkpoint['model_state_dict'].keys()

# Assuming you already loaded 'checkpoint' variable
state_dict = checkpoint['model_state_dict']

# Get the list of all layer names
layer_names = list(state_dict.keys())

# Print sample keys to verify
print(f"Total layers saved: {len(layer_names)}")
print("\n--- SAMPLE LAYERS ---")
for key in layer_names[:5]:
    print(key)

# Check specifically for your sub-models
print("\n--- CONTENT CHECK ---")
print(f"Has Audio Model? {any('audio_model' in k for k in layer_names)}")
print(f"Has Face Model?  {any('face_model' in k for k in layer_names)}")
print(f"Has Fusion?      {any('classifier' in k for k in layer_names)}")