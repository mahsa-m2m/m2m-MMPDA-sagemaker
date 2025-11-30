
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

class FusionModule(nn.Module):
    """
    including types of encoder w/ or w/o adapters and fusion methods
    """
    def __init__(self, args):
        super(FusionModule, self).__init__()
        self.fusion_type = args.fusion_type  # to indicate fusion type: "concat" or "cross2"
        self.multi = True  # multitask learning with multiple losses for audio and visual
        self.modalities = args.modalities
        self.combined_dim = len(self.modalities) * args.common_dim

        # self.vision_model = AU_GAZE_Affect7_MLP_MLP() # AU_GAZE_Affect7_MLP_MLP
        self.vision_model = AU_GAZE_Affect7_LSTM_MLP(bidirectional=False) # AU_GAZE_Affect7_MLP_MLP
        self.vision_projector = nn.Conv1d(args.v_dim, args.common_dim, kernel_size=1, padding=0, bias=False)

        self.audio_model = ResNet18_audio()
        # self.audio_model = ResNet18_audio_LSTM(bidirectional=True)
        self.audio_projector = nn.Conv1d(args.a_dim, args.common_dim, kernel_size=1, padding=0, bias=False)

        self.face_model = ResNet18_LSTM() # ResNet18_GRU()
        # self.face_model = ResNet18_face_LSTM(bidirectional=True)
        self.face_projector = nn.Conv1d(args.f_dim, args.common_dim, kernel_size=1, padding=0, bias=False)

        if self.fusion_type == "concat":
            self.fusion = SimpleConcat()
        elif self.fusion_type == 'transformer':
            self.fusion_layer = self.transformer_attention(args)
        elif self.fusion_type == 'senet':
            self.fusion_layer = self.se_fusion(args)
        elif self.fusion_type == 'mult':
            self.mult_fusion(args)
        else:
            raise Exception("Undefined fusion type!")

        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.combined_dim // 2, 2),
        ) # add one dim to store

    def se_fusion(self, params):
        fusion_nets = []
        for i in range(len(self.modalities)):
            fusion_nets.append(SELayer(params).to(params.device))
        return fusion_nets

    def transformer_attention(self, params):
        fusion_nets = []
        for i in range(len(self.modalities)):
            fusion_nets.append(TransformerEncoder(embed_dim=params.embed_dim,
                                                  num_heads=params.num_heads,
                                                  layers=params.layers,
                                                  attn_dropout=params.attn_dropout,
                                                  relu_dropout=params.relu_dropout,
                                                  res_dropout=params.res_dropout,
                                                  embed_dropout=params.embed_dropout,
                                                  attn_mask=params.attn_mask).to(params.device))
        return fusion_nets

    def get_network(self, embed_dim, params):
        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                num_heads=params.num_heads,
                                layers=params.mult_layer,
                                attn_dropout=params.attn_dropout,
                                relu_dropout=params.relu_dropout,
                                res_dropout=params.res_dropout,
                                embed_dropout=params.embed_dropout,
                                attn_mask=params.attn_mask).to(params.device)

    def mult_fusion(self, params):
        # 1. Temporal convolutional layers
        # self.proj_a = nn.Conv1d(params.common_dim, params.common_dim, kernel_size=1, padding=0, bias=False)
        # self.proj_v = nn.Conv1d(params.common_dim, params.common_dim, kernel_size=1, padding=0, bias=False)
        # self.proj_f = nn.Conv1d(params.common_dim, params.common_dim, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_f_with_a = self.get_network(params.common_dim, params)
        self.trans_f_with_v = self.get_network(params.common_dim, params)
    
        self.trans_a_with_f = self.get_network(params.common_dim, params)
        self.trans_a_with_v = self.get_network(params.common_dim, params)
    
        self.trans_v_with_f = self.get_network(params.common_dim, params)
        self.trans_v_with_a = self.get_network(params.common_dim, params)

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_f_mem = self.get_network(2*params.common_dim, params)
        self.trans_a_mem = self.get_network(2*params.common_dim, params)
        self.trans_v_mem = self.get_network(2*params.common_dim, params)

        # resdual block
        self.proj1 = nn.Linear(2*self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, 2*self.combined_dim)
        self.proj3 = nn.Linear(2*self.combined_dim, self.combined_dim)

    def transformer_attention(self, params):
        fusion_nets = []
        for i in range(len(self.modalities)):
            fusion_nets.append(TransformerEncoder(embed_dim=params.embed_dim,
                                                  num_heads=params.num_heads,
                                                  layers=params.layers,
                                                  attn_dropout=params.attn_dropout,
                                                  relu_dropout=params.relu_dropout,
                                                  res_dropout=params.res_dropout,
                                                  embed_dropout=params.embed_dropout,
                                                  attn_mask=params.attn_mask).to(params.device))
        return fusion_nets

    def forward(self, vision_behaviour, vision_face, audio_mel,  audio_wave):
        feature_list = []

        # Low-level Audio Feature extraction by ResNet
        al_logits, audio_mels = self.audio_model(audio_mel)
        # audio_mels = audio_mels.permute(0, 2, 1)
        audio_mels = self.audio_projector(audio_mels) #.squeeze(-1)

        # Low-level Vision Feature extraction by MLP 
        vl_logits, vision_behaviours = self.vision_model(vision_behaviour)
        vision_behaviours = vision_behaviours.permute(0, 2, 1)
        vision_behaviours = self.vision_projector(vision_behaviours)

        # Low-level Face Feature extraction by ResNet
        face_logits, face_feat = self.face_model(vision_face)
        # face_feat = face_feat.permute(0, 2, 1)
        face_feat = self.face_projector(face_feat)

        feature_list = [audio_mels, vision_behaviours, face_feat] # 7, 64, 32

        if self.fusion_type == "concat":
            # simple concatenation
            # audio_waves = self.AUDIO_TRANSFORMER(audio_waves)
            # vision_faces = self.VISION_TRANSFORMER(vision_faces)
            fused_output = torch.cat((audio_mels, vision_behaviours), dim=-1)
            fused_logit = self.classifier(fused_output)
        elif self.fusion_type == 'senet':
                # each modality is input into its senet module, the the outputs are concatenated for classification
                res = [m(x.unsqueeze(-1)) for x, m in zip(feature_list, self.fusion)]
                res_tensor = torch.cat(res, dim=1).squeeze(-1).squeeze(-1)  # torch.Size([8, 192])
                fused_logit = self.classifier(res_tensor)
        elif self.fusion_type == 'transformer':
            # transformer attention between x1-x2 or x1-x2-x3
            if len(feature_list) == 2:
                x1, x2 = feature_list
                x1, x2 = x1.permute(2, 0, 1), x2.permute(2, 0, 1)
                x1_to_x2 = self.fusion[0](x1, x2, x2).squeeze(0)
                x2_to_x1 = self.fusion[1](x2, x1, x1).squeeze(0)
                res_tensor = torch.cat([x1_to_x2, x2_to_x1], dim=-1)
            else:
                x1, x2, x3 = feature_list
                x1, x2, x3 = x1.permute(2, 0, 1), x2.permute(2, 0, 1), x3.permute(2, 0, 1)
                x1_to_x2 = self.fusion[0](x1, x2, x2).squeeze(0)
                x2_to_x3 = self.fusion[1](x2, x3, x3).squeeze(0)
                x3_to_x1 = self.fusion[2](x3, x1, x1).squeeze(0)
                res_tensor = torch.cat([x1_to_x2, x2_to_x3, x3_to_x1], dim=-1)
            fused_logit = self.classifier(res_tensor)
            fused_logit = torch.mean(fused_logit, 1)
        elif self.fusion_type == 'mult':
            proj_x_a, proj_x_v, proj_x_f = feature_list # [N, C, L]
            # proj_x_v = proj_x_v.transpose(1, 2)
            # proj_x_a = proj_x_a.transpose(1, 2)
            # proj_x_f = proj_x_f.transpose(1, 2)

            # Project the visual/audio/face features
            # proj_x_a = self.proj_a(proj_x_a)
            # proj_x_v = self.proj_v(proj_x_v)
            # proj_x_f = self.proj_f(proj_x_f)
            proj_x_a = proj_x_a.permute(2, 0, 1) # [L, N, C]
            proj_x_v = proj_x_v.permute(2, 0, 1)
            proj_x_f = proj_x_f.permute(2, 0, 1)

            # (V,A) --> F
            h_f_with_as = self.trans_f_with_a(proj_x_f, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_f_with_vs = self.trans_f_with_v(proj_x_f, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_fs = torch.cat([h_f_with_as, h_f_with_vs], dim=2)
            h_fs = self.trans_f_mem(h_fs)
            if type(h_fs) == tuple:
                h_fs = h_fs[0]
            last_h_f = last_hs = h_fs[-1]   # Take the last output for prediction

            # (F,V) --> A
            h_a_with_fs = self.trans_a_with_f(proj_x_a, proj_x_f, proj_x_f)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_fs, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]
            
            # (F,A) --> V
            h_v_with_fs = self.trans_v_with_f(proj_x_v, proj_x_f, proj_x_f)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_fs, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
            last_hs = torch.cat([last_h_f, last_h_a, last_h_v], dim=1)
            # A residual block
            # last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj = self.proj2(F.relu(self.proj1(last_hs), inplace=True))
            last_hs_proj += last_hs
            last_hs_proj = self.proj3(last_hs_proj)

            fused_logits = self.classifier(last_hs_proj)

        else:
            raise Exception("undefined fusion type")

        if self.multi:
            return fused_logits, vl_logits, face_logits, al_logits, [vision_behaviours, face_feat, audio_mels, last_hs]
        else:
            return fused_logits, None, None, None, None, [vision_behaviours, face_feat, audio_mels,last_hs]

# class LightweightFusionModule(nn.Module):
#     """
#     Simplified fusion for small datasets
#     Reduces params from 36M to ~5-10M
#     """
#     def __init__(self, args):
#         super(LightweightFusionModule, self).__init__()
#         self.multi = True
#         self.modalities = args.modalities
        
#         # CHANGE 1: Freeze pretrained encoders (if using pretrained weights)
#         # CHANGE 2: Use simpler behavioral model
#         self.vision_model = AU_GAZE_Affect7_LSTM_MLP(bidirectional=False)
        
#         # # OPTIONAL: Freeze vision model if not performing well
#         # for param in self.vision_model.parameters():
#         #     param.requires_grad = False
        
#         # Keep audio and face models
#         self.audio_model = ResNet18_audio()
#         self.face_model = ResNet18_LSTM()
        
#         # CHANGE 3: Simpler projections (no Conv1d, just Linear)
#         self.audio_proj = nn.Linear(args.a_dim, args.common_dim)
#         self.vision_proj = nn.Linear(args.v_dim, args.common_dim)
#         self.face_proj = nn.Linear(args.f_dim, args.common_dim)
        
#         # CHANGE 4: Replace complex transformers with simple attention
#         self.combined_dim = len(self.modalities) * args.common_dim
        
#         # Simple attention pooling instead of 6 transformers
#         self.attention = nn.MultiheadAttention(
#             embed_dim=args.common_dim,
#             num_heads=4,  # Reduced from potentially 8
#             dropout=0.3,
#             batch_first=True
#         )
        
#         # CHANGE 5: Simpler classifier with dropout
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),  # High dropout for regularization
#             nn.Linear(self.combined_dim, self.combined_dim // 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(self.combined_dim // 4, 2),
#         )

#     def forward(self, vision_behaviour, vision_face, audio_mel, audio_wave):
#         print("Forwarding LightFusion.............")
#         # Extract features from each modality
#         al_logits, audio_feats = self.audio_model(audio_mel)  # [B, a_dim, T]
#         vl_logits, vision_feats = self.vision_model(vision_behaviour)  # [B, 1, v_dim]
#         face_logits, face_feats = self.face_model(vision_face)  # [B, f_dim, T]
        
#         # Global average pooling to get fixed-size representations
#         audio_pooled = audio_feats.mean(dim=2)  # [B, a_dim]
#         vision_pooled = vision_feats.squeeze(1)  # [B, v_dim]
#         face_pooled = face_feats.mean(dim=2)  # [B, f_dim]
        
#         # Project to common dimension
#         audio_proj = self.audio_proj(audio_pooled)  # [B, common_dim]
#         vision_proj = self.vision_proj(vision_pooled)  # [B, common_dim]
#         face_proj = self.face_proj(face_pooled)  # [B, common_dim]
        
#         # Stack for attention: [B, 3, common_dim]
#         stacked = torch.stack([audio_proj, vision_proj, face_proj], dim=1)
        
#         # Simple cross-modal attention (replaces 6 transformers!)
#         attended, _ = self.attention(stacked, stacked, stacked)  # [B, 3, common_dim]
        
#         # Flatten for classification
#         fused = attended.reshape(attended.size(0), -1)  # [B, 3*common_dim]
        
#         # Classification
#         fused_logits = self.classifier(fused)
        
#         if self.multi:
#             return fused_logits, vl_logits, face_logits, al_logits, [vision_feats, face_feats, audio_feats, fused]
#         else:
#             return fused_logits, None, None, None, [vision_feats, face_feats, audio_feats, fused]

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
        
        # CHANGE 5: Simpler classifier with dropout
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

# ALTERNATIVE: Even simpler concatenation-based fusion (2-5M params)
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
