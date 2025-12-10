import torch
import torch.nn.functional as F
import numpy as np

class MultimodalExplainer:
    def __init__(self, model, target_layer_name='features'):
        self.model = model
        
        # storage for list of frames
        self.activations = [] 
        self.gradients = []
        
        # 1. Find Layer
        self.target_layer = self._find_target_layer(model.face_model, target_layer_name)
        
        # 2. Register "Sequence-Safe" Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def _find_target_layer(self, module, layer_name):
        """
        Robust search for the target layer.
        """
        print(f"DEBUG: Searching for target layer '{layer_name}'...")

        # 1. Direct match
        if hasattr(module, layer_name):
            target = getattr(module, layer_name)
            # If it's a list/sequential, return the last item (the deep features)
            if isinstance(target, torch.nn.Sequential) or isinstance(target, list):
                return target[-1]
            return target

        # 2. SPECIFIC FIX FOR ResNet18_LSTM with 'features'
        if hasattr(module, 'features'):
            print("DEBUG: Found 'features' container. Using the last block (Layer 7).")
            # The 'features' container has indices 0 to 7.
            # Index 7 is the standard 'layer4' equivalent (512 channels).
            return module.features[-1]

        # 3. Standard ResNet wrapper search (resnet, backbone, etc.)
        common_names = ['resnet', 'base_model', 'cnn', 'backbone', 'encoder']
        for name in common_names:
            if hasattr(module, name):
                submodule = getattr(module, name)
                # Recursively check inside the submodule
                if hasattr(submodule, layer_name):
                    return getattr(submodule, layer_name)[-1]
                # If submodule has 'features' (e.g. VGG style)
                if hasattr(submodule, 'features'):
                    return submodule.features[-1]

        # 4. Fallback: Recursive search by string
        for name, m in module.named_modules():
            if name.endswith(layer_name):
                return m[-1]

        # 5. FAILURE
        raise ValueError(f"Could not find layer '{layer_name}' in face_model. Structure uses 'features' list.")

    def save_activation(self, module, input, output):
        # If output is 4D (Batch*Time, C, H, W), it's already flattened
        # If output is called multiple times (Loop), we append to list
        if len(self.activations) > 0 and output.shape == self.activations[-1].shape:
             self.activations.append(output)
        else:
             self.activations = [output]

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if len(self.gradients) > 0 and grad.shape == self.gradients[-1].shape:
             self.gradients.append(grad)
        else:
             self.gradients = [grad]
             
    def explain(self, inputs, target_class_idx=None):
        v_beh, v_face, a_mel, a_wave = inputs
        
        # Clear storage before run
        self.activations = []
        self.gradients = []

        if not v_beh.requires_grad: v_beh.requires_grad = True
        
        # Forward & Backward
        self.model.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = self.model(v_beh, v_face, a_mel, a_wave)
            logits = outputs[0]
            if target_class_idx is None:
                target_class_idx = torch.argmax(logits, dim=1).item()
            score = logits[0, target_class_idx]
            score.backward()

        # --- PROCESS SEQUENCE DATA ---
        
        # 1. Stack list into tensor
        # If activations is a list of [1, 512, 7, 7] -> Stack to [T, 512, 7, 7]
        # If activations is one tensor [T, 512, 7, 7] -> Keep as is
        if isinstance(self.activations, list):
            fmaps = torch.cat(self.activations, dim=0)
        else:
            fmaps = self.activations

        if isinstance(self.gradients, list):
            grads = torch.cat(self.gradients, dim=0)
        else:
            grads = self.gradients

        # DEBUG: Check shapes
        print(f"DEBUG: CAM Gradients Shape: {grads.shape}") 
        # Should be (50, 512, 7, 7) or similar. If first dim is 1, that's the problem.

        # 2. Compute Grad-CAM
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * fmaps, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        face_cam = cam.data.cpu().numpy() # (Time, 1, H, W)
        
        # Calculate behavioral saliency
        beh_grad = v_beh.grad.data
        beh_saliency = torch.abs(beh_grad * v_beh.detach())
        beh_saliency = beh_saliency / (beh_saliency.max() + 1e-7)
        beh_saliency = beh_saliency.cpu().numpy()

        return face_cam, beh_saliency, target_class_idx