
import torch
import torch.nn as nn
import torchaudio
import torchvision
from models_comp.visual_model import cnn_face, AU_GAZE_Affect7_MLP_MLP, ResNet18_GRU, ResNet18_LSTM,AU_GAZE_Affect7_LSTM_MLP, ResNet18_face_LSTM
from models_comp.audio_model import ResNet18_audio, ResNet18_audio_LSTM
from torch.nn import functional as F

from modules.transformer import TransformerEncoder

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        
class SimpleConcat(nn.Module):
    def __init__(self, dim=1):
        super(SimpleConcat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)

class SELayer(nn.Module):
    """
    SE-concatenation: first concatenate all the embeddings from different modality then perform SE attention.
    reference: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, args):
        super(SELayer, self).__init__()
        # = args.reduction   #=16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(args.channel, args.channel // args.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(args.channel // args.reduction, args.channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

### our block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, Channel]
        b, c = x.size()
        y = self.fc(x).view(b, c)
        return x * y.expand_as(x) # Scale the features

class LightweightFusionModule(nn.Module):
    """
    Simplified fusion for small datasets
    Reduces params from 36M to ~5-10M
    """
    def __init__(self, args):
        super(LightweightFusionModule, self).__init__()
        self.multi = True
        self.modalities = args.modalities
        
        # Use simpler behavioral model
        self.vision_model = AU_GAZE_Affect7_LSTM_MLP(bidirectional=False)
        
        # # OPTIONAL: Freeze vision model if not performing well
        # for param in self.vision_model.parameters():
        #     param.requires_grad = False
        
        # Keep audio and face models
        self.audio_model = ResNet18_audio()
        self.face_model = ResNet18_LSTM()
        
        # Simpler projections (no Conv1d, just Linear)
        self.audio_proj = nn.Linear(args.a_dim, args.common_dim)
        self.vision_proj = nn.Linear(args.v_dim, args.common_dim)
        self.face_proj = nn.Linear(args.f_dim, args.common_dim)
        
        # Replace complex transformers with simple attention
        self.combined_dim = len(self.modalities) * args.common_dim
        
        self.layer_norm = nn.LayerNorm(args.common_dim)
        
        # Simple attention pooling instead of 6 transformers
        self.attention = nn.MultiheadAttention(
            embed_dim=args.common_dim,
            num_heads=4,  
            dropout=0.3,
            batch_first=True
        )
        
        # Simpler classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # High dropout for regularization
            nn.Linear(self.combined_dim, self.combined_dim // 4),
            # --- ADDED THIS ---
            nn.BatchNorm1d(self.combined_dim // 4), 
            # ----------------
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.combined_dim // 4, 2),
        )

    def forward(self, vision_behaviour, vision_face, audio_mel, audio_wave):
        print("Forwarding LightFusion.............")
        # Extract features from each modality
        al_logits, audio_feats = self.audio_model(audio_mel)  # [B, a_dim, T]
        vl_logits, vision_feats = self.vision_model(vision_behaviour)  # [B, 1, v_dim]
        face_logits, face_feats = self.face_model(vision_face)  # [B, f_dim, T]
        
        # Global average pooling to get fixed-size representations
        ################
        # audio_pooled = audio_feats.mean(dim=2)  # [B, a_dim]
        audio_pooled = audio_feats.max(dim=2)[0]  # Shape: [B, a_dim]

        vision_pooled = vision_feats.squeeze(1)  # [B, v_dim]
        ###############
        # face_pooled = face_feats.mean(dim=2)  # [B, f_dim]
        face_pooled = face_feats.max(dim=2)[0]    # Shape: [B, f_dim]
        
        # Project to common dimension
        audio_proj = self.audio_proj(audio_pooled)  # [B, common_dim]
        vision_proj = self.vision_proj(vision_pooled)  # [B, common_dim]
        face_proj = self.face_proj(face_pooled)  # [B, common_dim]
        
        # Stack for attention: [B, 3, common_dim]
        stacked = torch.stack([audio_proj, vision_proj, face_proj], dim=1)
        
        # Apply LayerNorm BEFORE Attention (Stabilizes gradients)
        normed_stacked = self.layer_norm(stacked)
        # Attention
        attn_out, _ = self.attention(normed_stacked, normed_stacked, normed_stacked)
        # Residual Connection
        # add the original 'stacked' to the 'attn_out'
        fused_features = stacked + attn_out    
        # Flatten
        fused = fused_features.reshape(fused_features.size(0), -1)
    
        
        # Classification
        fused_logits = self.classifier(fused)
        
        if self.multi:
            return fused_logits, vl_logits, face_logits, al_logits, [vision_feats, face_feats, audio_feats, fused]
        else:
            return fused_logits, None, None, None, [vision_feats, face_feats, audio_feats, fused]

class MinimalFusionModule(nn.Module):
    """Ultra-light fusion - just concatenate and classify"""
    def __init__(self, args):
        super(MinimalFusionModule, self).__init__()
        self.multi = True
        
        self.vision_model = AU_GAZE_Affect7_LSTM_MLP(bidirectional=False)
        self.audio_model = ResNet18_audio()
        self.face_model = ResNet18_LSTM()
        

        # Total input = a_dim + v_dim + f_dim (after pooling)
        total_dim = args.a_dim + args.v_dim + args.f_dim
        
        self.se_block = SEBlock(total_dim, reduction=16)

        # Simple MLP classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_dim), ### added batchnorm
            nn.Dropout(0.5),
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, vision_behaviour, vision_face, audio_mel, audio_wave):
        # Extract features
        al_logits, audio_feats = self.audio_model(audio_mel)
        vl_logits, vision_feats = self.vision_model(vision_behaviour)
        face_logits, face_feats = self.face_model(vision_face)
        
        # Pool to fixed size
        # audio_pooled = audio_feats.mean(dim=2)
        audio_pooled = audio_feats.max(dim=2)[0]  # Shape: [B, a_dim]

        vision_pooled = vision_feats.squeeze(1)
        
        # face_pooled = face_feats.mean(dim=2)
        face_pooled = face_feats.max(dim=2)[0]    # Shape: [B, f_dim]

        
        # Simple concatenation
        fused = torch.cat([audio_pooled, vision_pooled, face_pooled], dim=1)
        
        # Apply Attention Weighting
        fused = self.se_block(fused)

        # Classify
        fused_logits = self.classifier(fused)
        
        if self.multi:
            return fused_logits, vl_logits, face_logits, al_logits, [vision_feats, face_feats, audio_feats, fused]
        else:
            return fused_logits, None, None, None, [vision_feats, face_feats, audio_feats, fused]

class FusionModule(nn.Module):
    """
    2 Modalities: Vision (Behaviour) + Face
    Audio is removed.
    """
    def __init__(self, args):
        super(FusionModule, self).__init__()
        self.fusion_type = args.fusion_type 
        self.multi = True  
        # FORCE MODALITIES TO JUST V and F
        self.modalities = ['visual', 'face'] 
        self.combined_dim = len(self.modalities) * args.common_dim # Now 2 * 128 = 256

        # 1. VISION MODEL (Behavior)
        self.vision_model = AU_GAZE_Affect7_LSTM_MLP(bidirectional=False)
        self.vision_projector = nn.Conv1d(args.v_dim, args.common_dim, kernel_size=1, padding=0, bias=False)

        # 2. FACE MODEL
        self.face_model = ResNet18_LSTM() 
        self.face_projector = nn.Conv1d(args.f_dim, args.common_dim, kernel_size=1, padding=0, bias=False)

        # --- AUDIO MODEL ---
        # self.audio_model = ResNet18_audio()
        # self.audio_projector = nn.Conv1d(args.a_dim, args.common_dim, kernel_size=1, padding=0, bias=False)

        if self.fusion_type == "concat":
            self.fusion = SimpleConcat()
        elif self.fusion_type == 'mult':
            self.mult_fusion(args)
        else:
            raise Exception("Only 'mult' or 'concat' adapted for 2-modality in this snippet.")

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim // 2),
            nn.BatchNorm1d(self.combined_dim // 2), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                        
            nn.Linear(self.combined_dim // 2, 2),
        )

    def get_network(self, embed_dim, params):
        return TransformerEncoder(embed_dim=embed_dim,
                                num_heads=params.num_heads,
                                layers=params.mult_layer,
                                attn_dropout=params.attn_dropout,
                                relu_dropout=params.relu_dropout,
                                res_dropout=params.res_dropout,
                                embed_dropout=params.embed_dropout,
                                attn_mask=params.attn_mask).to(params.device)

    def mult_fusion(self, params):
        # ADAPTED FOR 2 MODALITIES
        
        # Vision attending to Face
        self.trans_v_with_f = self.get_network(params.common_dim, params)
        # Face attending to Vision
        self.trans_f_with_v = self.get_network(params.common_dim, params)
    
        # Input dim is just common_dim
        self.trans_v_mem = self.get_network(params.common_dim, params)
        self.trans_f_mem = self.get_network(params.common_dim, params)

        # Residual block
        # size is 2 * common_dim
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        # self.proj3 = nn.Linear(self.combined_dim, self.combined_dim)

    def forward(self, vision_behaviour, vision_face, audio_mel=None, audio_wave=None):
        # --- 1. FEATURE EXTRACTION ---
        
        # Vision (Behavior)
        vl_logits, vision_behaviours = self.vision_model(vision_behaviour)
        vision_behaviours = vision_behaviours.permute(0, 2, 1)
        vision_behaviours = self.vision_projector(vision_behaviours) # [B, Dim, T]

        # Face
        face_logits, face_feat = self.face_model(vision_face)
        face_feat = self.face_projector(face_feat) # [B, Dim, T]

        # Audio REMOVED
        al_logits = None
        audio_mels = None 

        feature_list = [vision_behaviours, face_feat]

        # --- 2. FUSION ---
        if self.fusion_type == 'mult':
            proj_x_v, proj_x_f = feature_list 
            
            # Permute for Transformer [Sequence, Batch, Dim]
            proj_x_v = proj_x_v.permute(2, 0, 1) 
            proj_x_f = proj_x_f.permute(2, 0, 1)

            # (Vision, Face) --> Face
            # Q=Face, K=Vision, V=Vision
            h_f_with_vs = self.trans_f_with_v(proj_x_f, proj_x_v, proj_x_v)
            h_fs = self.trans_f_mem(h_f_with_vs)
            if isinstance(h_fs, tuple): h_fs = h_fs[0]
            last_h_f = h_fs[-1] 

            # (Face, Vision) --> Vision
            # Q=Vision, K=Face, V=Face
            h_v_with_fs = self.trans_v_with_f(proj_x_v, proj_x_f, proj_x_f)
            h_vs = self.trans_v_mem(h_v_with_fs)
            if isinstance(h_vs, tuple): h_vs = h_vs[0]
            last_h_v = h_vs[-1]
        
            # Concatenate: [Batch, 2*Dim]
            last_hs = torch.cat([last_h_v, last_h_f], dim=1)
            
            # Residual Block
            last_hs_proj = self.proj2(F.relu(self.proj1(last_hs), inplace=True))
            last_hs_proj += last_hs
            
            fused_logits = self.classifier(last_hs_proj)

        elif self.fusion_type == "concat":
            fused_output = torch.cat((vision_behaviours, face_feat), dim=1)
            # Global Average Pooling to get [B, Dim]
            fused_output = torch.mean(fused_output, dim=2) 
            fused_logit = self.classifier(fused_output)

        else:
            raise Exception("undefined fusion type")

        if self.multi:
            # al_logits is None
            return fused_logits, vl_logits, face_logits, None, [vision_behaviours, face_feat]
        else:
            return fused_logits, None, None, None, None, [vision_behaviours, face_feat]

if __name__ == '__main__':

    # for testing :
    # model = FusionModule("concat", num_encoders=2, adapter=True, adapter_type="efficient_conv").cuda()
    model = FusionModule("concat", num_encoders=2, adapter=True, adapter_type="efficient_conv")
    print(model)
    # inp = torch.rand(8, 20601).cuda()  # batch_size = 8
    inp = torch.rand(8, 20601)
    # vis = torch.rand(8, 64, 3, 160, 160).cuda()
    vis = torch.rand(8, 64, 3, 160, 160)
    out = model(inp, vis)
    print(out.shape)

"""
fusion types:

1- simple concatenation of the final outputs from audio and face models
2- concatenation between each encoders and final concatenation 
"""
