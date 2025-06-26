import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from ultralytics.nn.modules.block import Conv, Bottleneck
from torchvision.models import vit_b_16
import numpy as np
import math
from PIL import Image
from ultralytics.nn.modules.block import ABlock, C3k

# __all__ = (
#     "DFL",
#     "HGBlock",
#     "HGStem",
#     "SPP",
#     "SPPF",
#     "C1",
#     "C2",
#     "C3",
#     "C2f",
#     "C2fAttn",
#     "ImagePoolingAttn",
#     "ContrastiveHead",
#     "BNContrastiveHead",
#     "C3x",
#     "C3TR",
#     "C3Ghost",
#     "GhostBottleneck",
#     "Bottleneck",
#     "BottleneckCSP",
#     "Proto",
#     "RepC3",
#     "ResNetLayer",
#     "RepNCSPELAN4",
#     "ELAN1",
#     "ADown",
#     "AConv",
#     "SPPELAN",
#     "CBFuse",
#     "CBLinear",
#     "C3k2",
#     "C2fPSA",
#     "C2PSA",
#     "RepVGGDW",
#     "CIB",
#     "C2fCIB",
#     "Attention",
#     "PSA",
#     "SCDown",
#     "TorchVision",
#     "AAttn",
#     "ABlock",
#     "A2C2f",
#     "TemplateAttn",
#     "TemplateBlock",
#     "A2C2fTemplate"
# )

class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)  # c2 and ec should be same
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0
        self.c1 = c1
        self.c2 = c2

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape
        guide = self.gl(guide)
        
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)

class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))
      
class C2f_v2v_Attn(nn.Module):
    """C2f module with an additional attn module, for v2v (replace clip as dinov2)."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=768, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()
        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)

        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))

        text = text.reshape(bs, -1, text.shape[-1])

        return x * self.scale + text

class TemplateAttentionPooling(nn.Module):
    def __init__(self, hidden_size=768, proj_size=512):
        super().__init__()
        self.attention_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.proj_layer = nn.Linear(hidden_size, proj_size)
        
        # Initialize
        nn.init.xavier_uniform_(self.attention_query)
        self.scale = hidden_size ** -0.5
        
    def forward(self, patch_tokens):
        """
        Input
            patch_tokens: [batch_size, 50, hidden_size] (CLIP patch features)
        Output:
            pooled_feature: [batch_size, hidden_size] (Same shape as CLS token)
        """
        batch_size = patch_tokens.shape[0]
        
        # patches projection
        K = self.key_proj(patch_tokens)    # [B, 50, D]
        V = self.value_proj(patch_tokens)  # [B, 50, D]
        
        # expand query to match batch size
        Q = self.attention_query.expand(batch_size, -1, -1)  # [B, 1, D]
        
        # Attention Score
        # attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, 1, 50]
        attn_scores = torch.einsum("bqd,bkd->bqk", Q, K) * self.scale  # [B, 1, 50]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, 50]

        # Sum up the weighted patches
        # pooled = torch.matmul(attn_weights, V)  # [B, 1, D]
        # pooled = pooled.squeeze(1)  # [B, D]
        pooled = torch.einsum("cqp,cpd->cd", attn_weights, V)

        # Normalize
        pooled = self.layer_norm(pooled)
        
        proj_pooled = self.proj_layer(pooled)
        
        # return pooled, attn_weights.squeeze(1)  # [B, D], [B, 50]
        return {
            "pooled_feature": pooled,
            "attn_weights": attn_weights.squeeze(1),
            "pooled_feature_proj": proj_pooled
        }

class MultiLevelTemplateAttentionPooling(nn.Module):
    def __init__(self, hidden_size=768, proj_size=512, num_patches=50, num_levels=3):
        super().__init__()
        self.attention_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.proj_layer = nn.Linear(hidden_size, proj_size)
        
        # add level embeddings
        self.level_embeddings = nn.Parameter(torch.randn(num_levels, 1, hidden_size))
        
        # Initialize
        nn.init.xavier_uniform_(self.attention_query)
        self.scale = hidden_size ** -0.5
        
        self.num_patches = num_patches
        self.num_levels = num_levels
        
    def forward(self, patch_tokens_list):
        """
        Input:
            patch_tokens_list: List of tensors, each of shape [batch_size, num_patches, hidden_size]
        """
        # Concat all patch tokens
        batch_size = patch_tokens_list[0].shape[0]
        patch_tokens = torch.cat(patch_tokens_list, dim=1)  # [B, num_patches*L, D]
        
        # Add level embeddings
        level_embed = torch.cat([self.level_embeddings[i].expand(batch_size, self.num_patches, -1) for i in range(self.num_levels)], dim=1)
        patch_tokens = patch_tokens + level_embed

        K = self.key_proj(patch_tokens)    # [B, num_patches*L, D]
        V = self.value_proj(patch_tokens)  # [B, num_patches*L, D]
        
        Q = self.attention_query.expand(batch_size, -1, -1)  # [B, 1, D]
        
        attn_scores = torch.einsum("bqd,bkd->bqk", Q, K) * self.scale  # [B, 1, num_patches*L]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, num_patches*L]
        
        pooled = torch.einsum("cqp,cpd->cd", attn_weights, V)
        
        pooled = self.layer_norm(pooled)
        proj_pooled = self.proj_layer(pooled)
        
        return {
            "pooled_feature": pooled,
            "attn_weights": attn_weights.squeeze(1),
            "pooled_feature_proj": proj_pooled
        }
        
class MultiHeadTemplateAttentionPooling(nn.Module):
    def __init__(self, hidden_size=768, proj_size=512, num_patches=50, num_levels=3, num_heads=4, use_mlp=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.num_levels = num_levels
        self.use_mlp = use_mlp

        # Learnable query per head
        self.attention_query = nn.Parameter(torch.randn(num_heads, 1, hidden_size))

        # Key/Value projections
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # Learnable temperature (scaling factor)
        self.scale = nn.Parameter(torch.tensor(hidden_size ** -0.5))

        # Level embeddings
        self.level_embeddings = nn.Parameter(torch.randn(num_levels, 1, hidden_size))

        # Normalization and projection
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.proj_layer = nn.Linear(hidden_size, proj_size)

        # Optional MLP head for richer embeddings
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.LayerNorm(proj_size),
                nn.Linear(proj_size, proj_size),
                nn.GELU(),
                nn.Linear(proj_size, proj_size)
            )

        # Init
        nn.init.xavier_uniform_(self.attention_query)

    def forward(self, patch_tokens_list):
        """
        Input:
            patch_tokens_list: List of [B, num_patches, hidden_size] tensors (one per level)
        Output:
            Dict with:
              - pooled_feature: [B, hidden_size]
              - pooled_feature_proj: [B, proj_size]
              - attn_weights: [B, H, N]
        """
        batch_size = patch_tokens_list[0].shape[0]
        patch_tokens = torch.cat(patch_tokens_list, dim=1)  # [B, L*N, D]

        # Add level embeddings
        level_embed = torch.cat([
            self.level_embeddings[i].expand(batch_size, self.num_patches, -1)
            for i in range(self.num_levels)
        ], dim=1)
        patch_tokens = patch_tokens + level_embed  # [B, L*N, D]

        # Project keys and values
        K = self.key_proj(patch_tokens)  # [B, N, D]
        V = self.value_proj(patch_tokens)  # [B, N, D]

        # Expand queries
        Q = self.attention_query.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, H, 1, D]
        K = K.unsqueeze(1)  # [B, 1, N, D]
        V = V.unsqueeze(1)  # [B, 1, N, D]

        # Attention: [B, H, 1, D] x [B, 1, N, D] → [B, H, 1, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, 1, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, 1, N]

        # Aggregate: [B, H, 1, N] x [B, 1, N, D] → [B, H, 1, D]
        pooled = torch.matmul(attn_weights, V)  # [B, H, 1, D]
        pooled = pooled.squeeze(2)  # [B, H, D]

        # Combine heads (mean or concat + proj)
        pooled = pooled.mean(dim=1)  # [B, D]
        pooled = self.layer_norm(pooled)
        proj_pooled = self.proj_layer(pooled)  # [B, proj_size]

        if self.use_mlp:
            proj_pooled = self.mlp(proj_pooled)

        return {
            "pooled_feature": pooled,               # Pre-projection
            "pooled_feature_proj": proj_pooled,     # Final embedding
            "attn_weights": attn_weights.squeeze(2) # [B, H, N]
        }

class C2fAttnWithPatch(nn.Module):
    def __init__(self, c1, c2, n=1, ec=128, nh=1, patch_dim=196, patch_tokens=49, shortcut=False, g=1, e=0.5):
        """
        patch_dim: Feature dimension for each patch
        patch_tokens: Number of patches (e.g., 7x7=49)
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        
        # Modify attention block to handle patch input
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=patch_dim, ec=ec, nh=nh)
        
        # Optional: Add positional encoding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_tokens, patch_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x, patch_tokens):
        """
        x: Standard feature map input [B, C, H, W]
        patch_tokens: Patch features [B, N, D]
                     where N is number of patches, D is dimension per patch
        """
        # Add positional encoding
        patch_tokens = patch_tokens + self.pos_embed
        
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], patch_tokens))
        return self.cv2(torch.cat(y, 1))
    
class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch, c3, embed):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension."""
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

        # c3 = 256
        # embed = 512
        # ch = 

    def forward(self, x, vp):
        """Process input features and visual prompts to generate enhanced embeddings."""
        # x[0].shape = (bs, 256, 80, 80)
        # x[1].shape = (bs, 512, 40, 40)
        # x[2].shape = (bs, 512, 20, 20)
        # vp.shape = (bs, max_num_class, 80, 80)
        y = [self.cv2[i](xi) for i, xi in enumerate(x)] 
        # y[0].shape = (bs, 128, 80, 80)
        # y[1].shape = (bs, 128, 80, 80)
        # y[2].shape = (bs, 128, 80, 80)
        y = self.cv4(torch.cat(y, dim=1))
        # y.shape = (bs, 16, 80, 80)

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        # x[0].shape = (bs, 128, 80, 80)
        # x[1].shape = (bs, 128, 80, 80)
        # x[2].shape = (bs, 128, 80, 80)
        x = self.cv3(torch.cat(x, dim=1)) 
        # x.shape = (bs, 512, 80, 80)

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)
        # x.shape = (bs, 512, 6400)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)
        # vp.shape = (10, 1, 80, 80)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))
        # y.shape = (10, 16, 80, 80)

        y = y.reshape(B, Q, self.c, -1)
        # y.shape = (1, 10, 16, 6400)
        vp = vp.reshape(B, Q, 1, -1)
        # vp.shape = (1, 10, 1, 6400)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        # torch.finfo(y.dtype).min = -1.7014117e+38, the min value of float32
        # Given the maximum value of the negative number to ensure that the outcome of softmax is 0.
        
        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class PATCH_EMBEDDING_SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for patch embedding feature enhancement."""

    def __init__(self, c3=256, embed_dim=768, final_embed=512):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension."""
        super().__init__()
        
        # y Activation Branch - 16 Channels
        self.y_light_branch = nn.Sequential(
                    nn.Conv2d(embed_dim*3, 1024, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(1024, 576, kernel_size=1),
                    nn.PixelShuffle(upscale_factor=6),
                    nn.Upsample(size=(80, 80), mode='bilinear', align_corners=False)
                )
        
        # x Semantic Branch - 512 Channels
        self.x_heavy_branch = nn.Sequential(
            nn.Conv2d(embed_dim*3, 1024, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Upsample(size=(80, 80), mode='bilinear', align_corners=False)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, final_embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, embeddings, vp=None):
        """Process input features and visual prompts to generate enhanced embeddings."""
        
        self.device = next(self.parameters()).device
        
        if vp is None:
            vp = torch.zeros(embeddings.shape[0], 1, 80, 80)
        
        if embeddings.device!=self.device:
            embeddings = embeddings.to(self.device)
        if vp.device!=self.device:
            vp = vp.to(self.device)
        
        if vp.dim() == 5:
            # [B, S, NUM_LAYERS (3), PATCH_NUM, EMBEDDING_DIM]
            # Merge the batch and sequence dimensions
            vp = vp.squeeze(0)

        if vp.dim() == 4:
            B, Q, H, W = vp.shape
        elif vp.dim() == 3:
            Q, H, W = vp.shape
            B = 1
            vp = vp.unsqueeze(0)
        else:
            raise ValueError(f"Input tensor 'vp' shape should be >=3 dimensions, but got {vp.dim()} dimensions.")
                    
        if not isinstance(embeddings, list) and not isinstance(embeddings, torch.Tensor):
            raise TypeError("Unsupported type for embeddings. Must be list or torch.Tensor.")
        
        if isinstance(embeddings, list):
            
            assert len(embeddings) == 3, "The length of embeddings list must be 3. We only take 3 layers from those image encoder."
            
            embeddings = torch.stack(tensors=embeddings, dim=1)
            
        if embeddings.dim() == 5:
            # [B, S, NUM_LAYERS (3), PATCH_NUM, EMBEDDING_DIM]
            # Merge the batch and sequence dimensions
            embeddings = embeddings.squeeze(0)
        elif embeddings.dim() == 4:
            # [B, NUM_LAYERS (3), PATCH_NUM, EMBEDDING_DIM]
            pass
        elif embeddings.dim() == 3:
            # [NUM_LAYERS (3), PATCH_NUM, EMBEDDING_DIM]
            # Add batch dimension
            embeddings = embeddings.unsqueeze(0)
        else:
            raise ValueError("Input tensor must have at least 3 dimensions.")

        BS, NUM_LAYERS, PATCH_NUM, EMBEDDING_DIM = embeddings.shape
        
        if embeddings.shape[2] & 1 == 1:  # If number is odd, means this embedding has CLS token, remove it
            patch_emb = embeddings[:, :, 1:, :]  # [1, 256, 768]
        else: # No CLS token
            patch_emb = embeddings
        grid_size = int(math.sqrt(patch_emb.shape[2]))
        
        spatial_feat = patch_emb.reshape(BS, 3, grid_size, grid_size, -1)
        spatial_feat = spatial_feat.permute(0, 1, 4, 2, 3)
        spatial_feat = spatial_feat.reshape(BS, -1, grid_size, grid_size)
        
        y = self.y_light_branch(spatial_feat)
        x = self.x_heavy_branch(spatial_feat)
        
        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)
        
        vp_conv = self.cv5(vp)  # F_v
        cat_y_vp = torch.cat((y, vp_conv), dim=1) # Concat F_v and F_I
        y = self.cv6(cat_y_vp)

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min

        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)

    def reshape_patch_embeddings(self, embeddings, grid_size=16):
        """Reshape patch embeddings to spatial feature map."""
        B, N, D = embeddings.shape  # [1, 257, 768]
        
        # Remove CLS token
        patch_tokens = embeddings[:, 1:, :]  # [1, 256, 768]
        
        # Reshape to spatial feature map, like the yolo one
        spatial_features = patch_tokens.reshape(B, grid_size, grid_size, D)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # [1, 768, 16, 16]
        
        return spatial_features

class ResidualPixelShuffle(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone
        self.channel_adjust = nn.Conv2d(768, 576, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=6)
        
        self.skip_conv = nn.Conv2d(768, 64, kernel_size=1)
        self.skip_ps = nn.PixelShuffle(upscale_factor=2)
        self.skip_up = nn.Upsample(size=(80, 80), mode='bilinear', align_corners=False)
        
        self.fusion = nn.Conv2d(16+16, 16, kernel_size=3, padding=1)
        self.final_adjust = nn.Upsample(size=(80, 80), mode='bilinear', align_corners=False)
        self.norm = nn.LayerNorm([16, 80, 80])
        self.act = nn.GELU()
    
    def forward(self, x):
        main = self.channel_adjust(x)  # (1, 576, 14, 14)
        main = self.pixel_shuffle(main)  # (1, 16, 84, 84)
        main = self.final_adjust(main)  # (1, 16, 80, 80)
        
        skip = self.skip_conv(x)  # (1, 64, 14, 14)
        skip = self.skip_ps(skip)  # (1, 16, 28, 28)
        skip = self.skip_up(skip)  # (1, 16, 80, 80)
        
        out = torch.cat([main, skip], dim=1)  # (1, 32, 80, 80)
        out = self.fusion(out)  # (1, 16, 80, 80)
        out = self.act(self.norm(out))  # (1, 16, 80, 80)
        
        return out

class TemplateMatchingHead(nn.Module):
    """
    使用全局池化的方法做 template matching
    模板跟輸入影像的特徵皆經過相同的 backbone 提取
    """
    def __init__(self, embed_dims: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        
    def forward(self, input_feat, template_feat):
        """
        input_feat: 輸入影像特徵，形狀 (B, C, H, W)
        template_feat: 模板影像特徵，形狀 (B, C, H_t, W_t)
        """
        # 對輸入影像特徵做 BatchNorm
        input_norm = self.norm(input_feat)
        
        # 對 template 特徵進行全局平均池化，得到 (B, C)
        template_vec = F.adaptive_avg_pool2d(template_feat, (1, 1)).squeeze(-1).squeeze(-1)
        # 對模板向量做 L2 正規化
        template_vec = F.normalize(template_vec, p=2, dim=-1)
        
        # 此外也可以對輸入影像特徵每個位置做正規化（依據需求）
        # 計算每個空間位置與模板向量的內積：使用 einsum 來做點乘
        # einsum 的 "bchw,bc->bhw" 表示對每個 (h, w) 位置計算 dot product
        similarity = torch.einsum("bchw,bc->bhw", input_norm, template_vec)
        
        # 將相似度乘上尺度因子並加上 bias
        logits = similarity * self.logit_scale.exp() + self.bias
        return logits

class SpatialTemplateMatchingHead(nn.Module):
    """
    使用保留模板空間資訊的方法做 template matching
    以模板特徵作為卷積核，對輸入影像特徵做 cross-correlation
    """
    def __init__(self, embed_dims: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        
    def forward(self, input_feat, template_feat):
        """
        input_feat: 輸入影像特徵，形狀 (B, C, H, W)
        template_feat: 模板影像特徵，形狀 (B, C, H_t, W_t)
        """
        input_norm = self.norm(input_feat)
        # 對模板做 L2-normalize，但要在 channel 維度上做（保留空間資訊）
        template_norm = F.normalize(template_feat, p=2, dim=1)
        
        B = input_norm.shape[0]
        sim_list = []
        # 對每一個 batch 分別進行卷積匹配
        for i in range(B):
            # 將模板視為卷積核，形狀需為 (out_channels, in_channels, H_t, W_t)
            # 這裡假設模板只有一組核，因此先取出第 i 個 batch，並擴展維度
            kernel = template_norm[i:i+1]  # (1, C, H_t, W_t)
            # 對應的輸入影像：形狀 (1, C, H, W)
            inp = input_norm[i:i+1]
            # 計算 cross-correlation，注意卷積內部會反轉 kernel，但對於相似度計算可以忽略
            sim = F.conv2d(inp, kernel)
            sim = sim * self.logit_scale.exp() + self.bias
            sim_list.append(sim)
            
        # 將所有 batch 組合起來
        logits = torch.cat(sim_list, dim=0)
        return logits

class TemplateAttn(nn.Module):
    """
    Template Attention module using cross attention to learn features from template images.
    
    Attributes:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        area (int): Number of areas for feature map division (for compatibility with original area attention)

    Methods:
        forward: Applies attention with template tensor to input tensor.
        
    Examples:
        >>> attn = TemplateAttn(dim=256, num_heads=8)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> template = torch.randn(1, 256, 16, 16)
        >>> out = attn(x, template)
        torch.Size([1, 256, 32, 32])
    """
    
    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Projection layers for query, key, value
        self.q_proj = Conv(dim, dim, 1, act=False)
        self.k_proj = Conv(dim, dim, 1, act=False)
        self.v_proj = Conv(dim, dim, 1, act=False)
        
        # Output projection
        self.proj = Conv(dim, dim, 1, act=False)
        
        # Position-aware convolution (similar to PE in original AAttn)
        self.pe = Conv(dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x, template):
        """
        Forward pass for Template Attention.
        
        Attributes:
            x (torch.Tensor): Input features [B, C, H, W]
            template (torch.Tensor): Template features [B, C, H', W']
            
        Returns:
            torch.Tensor: Attention output [B, C, H, W]
        """
        B, C, H, W = x.shape
        B_t, C_t, H_t, W_t = template.shape
        
        # Compute query, key, value
        q = self.q_proj(x).flatten(2).transpose(1, 2)  # [B, H*W, C]
        k = self.k_proj(template).flatten(2).transpose(1, 2)  # [B, H'*W', C]
        v = self.v_proj(template).flatten(2).transpose(1, 2)  # [B, H'*W', C]
        
        # Reshape to multi-head format
        q = q.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, H*W, head_dim]
        k = k.reshape(B_t, H_t*W_t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, H'*W', head_dim]
        v = v.reshape(B_t, H_t*W_t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, H'*W', head_dim]

        # Calculate attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, H*W, H'*W']
        attn = attn.softmax(dim=-1)
        
        # Apply attention weights
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, H*W, C)  # [B, H*W, C]
        out = out.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        # Apply output projection and position-aware convolution
        out = self.proj(out)
        out = out + self.pe(v.permute(0, 2, 1, 3).reshape(B_t, C_t, H_t, W_t))
        
        return out


class TemplateBlock(nn.Module):
    """
    Template Block module combining cross attention and feed-forward network with residual connections.
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        mlp_ratio (float): Expansion ratio for MLP hidden dimension
        area (int): Number of areas for feature map division
    """
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()
        # Template attention module
        self.attn = TemplateAttn(dim, num_heads=num_heads, area=area)
        
        # Feed-forward network (MLP)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize module weights"""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, template):
        """
        Forward pass for Template Block
        
        Args:
            x (torch.Tensor): Input features [B, C, H, W]
            template (torch.Tensor): Template features [B, C, H', W']
            
        Returns:
            torch.Tensor: Output features [B, C, H, W]
        """
        # Apply template attention and first residual connection
        x = x + self.attn(x, template)
        
        # Apply feed-forward network and second residual connection
        x = x + self.mlp(x)
        
        return x


class A2C2fTemplate(nn.Module):
    """
    A2C2f variant using template cross attention instead of original area attention.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of TemplateBlock modules to stack
        area (int): Number of areas for feature map division (for compatibility)
        residual (bool): Whether to use residual connection
        mlp_ratio (float): Expansion ratio for MLP hidden dimension
        e (float): Channel expansion ratio for hidden channels
        num_heads (int): Number of attention heads
    """
    def __init__(self, c1, c2, n=1, area=1, residual=True, mlp_ratio=1.2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of TemplateBlock must be a multiple of 32"
        
        # Basic convolution layers
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        
        # Scaling parameter for residual connection
        # self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if residual else None
        
        # TemplateBlock module list
        self.m = nn.ModuleList(
            TemplateBlock(c_, c_ // 32, mlp_ratio, area) for _ in range(n)
        )

    def forward(self, x, template):
        """
        Forward pass for A2C2fTemplate
        
        Args:
            x (torch.Tensor): Input features [B, C, H, W]
            template (torch.Tensor): Template features [B, C, H', W']
            
        Returns:
            torch.Tensor: Output features [B, C, H, W]
        """
        # Initial feature processing
        y = [self.cv1(x)]
        
        # Apply TemplateBlock modules
        for m in self.m:
            y.append(m(y[-1], template))
        
        # Concatenate all features and apply final convolution
        y = self.cv2(torch.cat(y, 1))
        
        # If using residual connection, apply gamma parameter
        # if self.gamma is not None:
        #     return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y

class A2C2f_Template_MaxSigmoidAttn(nn.Module):
    """
    改進的 Area-Attention C2f 模組，能夠接受來自 template image 的特徵作為指導
    
    Args:
        c1 (int): 輸入通道數
        c2 (int): 輸出通道數
        n (int): A2Block 或 C3k 模組的堆疊數量
        a2 (bool): 是否使用 area attention blocks 若為 False則使用 C3k blocks
        area (int): 將特徵圖分割的區域數量
        residual (bool): 是否使用帶有可學習 gamma 參數的殘差連接
        mlp_ratio (float): MLP 隱藏維度的擴展比率
        e (float): 隱藏通道的通道擴展比率
        g (int): 分組卷積的組數
        shortcut (bool): C3k 塊中是否使用捷徑連接
        template_dim (int): template 特徵的維度
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, 
                 mlp_ratio=2.0, e=0.5, g=1, shortcut=True, template_dim=None, ec=128, nh=1):
        super().__init__()
        c_ = int(c2 * e)  # 隱藏通道
        assert c_ % 32 == 0, "ABlock 的維度必須是 32 的倍數"

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        
        # self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        
        # 加入 template 指導機制
        template_dim = template_dim or c_
        self.template_attn = MaxSigmoidAttnBlock(c_, c_, gc=template_dim, ec=ec, nh=nh)
        
        # 使用 ABlock 或 C3k
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x, template_guide):
        """
        前向傳播
        
        Args:
            x (torch.Tensor): 輸入特徵
            template_guide (torch.Tensor): 來自 template 的指導特徵
            
        Returns:
            torch.Tensor: 輸出特徵
        """
        y = [self.cv1(x)]
        
        # 對每個模組應用 ABlock 或 C3k
        for m in self.m:
            y.append(m(y[-1]))

        # 添加 template 指導
        y.append(self.template_attn(y[-1], template_guide))

        # 連接所有特徵並應用最終卷積
        y = self.cv2(torch.cat(y, 1))
        
        # 如果使用殘差連接，應用 gamma 参數
        # if self.gamma is not None:
        #     return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y

class FeatureAlignmentModel(nn.Module):
    def __init__(self, input_dim=576, output_dim=512):
        super(FeatureAlignmentModel, self).__init__()
        
        # Define layer
        self.backbone_c2f_align_linear_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.SiLU()
        )
        
        # 可以根據需要添加更多層
        # 例如: 物件檢測或分割的輸出層
        # self.classifier = nn.Linear(output_dim, num_classes)  # 如果需要分類
        # 或者用於分割的輸出層
        # self.segmentation_head = nn.Conv2d(output_dim, num_classes, kernel_size=1)
        
    def forward(self, x):
        # 處理輸入特徵
        # 假設 x 來自骨幹網路，形狀為 [batch_size, input_dim]
        
        # 特徵對齊
        original_shape = x.shape
        
        if len(original_shape) > 2:
            # 如果輸入是多維的 (例如 [batch_size, seq_len/height*width, channels])
            batch_size = original_shape[0]
            x_reshaped = x.view(-1, original_shape[-1])  # 重塑為 [batch_size*seq_len, channels]
            
            x_aligned = self.backbone_c2f_align_linear_layer(x_reshaped)
            
            # 重塑回原始維度，但最後一個維度現在是 output_dim
            new_shape = list(original_shape)
            new_shape[-1] = x_aligned.shape[-1]
            x_aligned = x_aligned.view(*new_shape)
        else:
            # 如果輸入是 [batch_size, channels]
            x_aligned = self.backbone_c2f_align_linear_layer(x)
        
        # 可以根據需要添加後續處理
        # 例如用於分類的輸出:
        # output = self.classifier(x_aligned)
        
        return x_aligned