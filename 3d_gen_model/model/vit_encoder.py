import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------------------------------------------------------
# Sinusoidal + Learnable Positional Embedding
# ------------------------------------------------------------------
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, max_positions=66, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim

        # Create sinusoidal embeddings
        position = torch.arange(max_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(max_positions, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        # Learnable scale and bias
        self.scale = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, seq_len, device):
        pos_embed = self.pe[:seq_len].unsqueeze(0).to(device)  # (1, seq_len, embed_dim)
        pos_embed = pos_embed * self.scale + self.bias
        return pos_embed


# ------------------------------------------------------------------
# DropPath (Stochastic Depth)
# ------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ------------------------------------------------------------------
# Multi-Head Self Attention
# ------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1, qkv_bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.temperature = nn.Parameter(torch.ones(1) * self.scale)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ------------------------------------------------------------------
# FeedForward (GLU variant)
# ------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * self.act(gate)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ------------------------------------------------------------------
# Transformer Block
# ------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_ratio, dropout)
        self.ls1 = nn.Parameter(torch.ones(embed_dim) * 1e-4)
        self.ls2 = nn.Parameter(torch.ones(embed_dim) * 1e-4)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ls2 * self.mlp(self.norm2(x)))
        return x


# ------------------------------------------------------------------
# Main DiffPoint ViT Encoder
# ------------------------------------------------------------------
class DiffPoint_ViT_Encoder(nn.Module):
    def __init__(self, embed_dim=768, num_patches=32, depth=21,
                 num_heads=16, mlp_ratio=4.5, dropout=0.1, drop_path_rate=0.2):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.seq_length = num_patches + 2  # patches + image + time
        self.depth = depth

        # ---- Mock modules (replace with your versions) ----
        from clip_attention import SAttention
        from pointMae import PointMAE
        from time_embedding import TimeEmbedding

        self.clip_att = SAttention()
        self.point_mae = PointMAE()
        self.time_embed = TimeEmbedding(embed_dim=embed_dim)

        # Positional embedding
        self.pos_embed = SinusoidalPositionEmbedding(max_positions=self.seq_length, embed_dim=embed_dim)

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])

        self.input_norm = nn.LayerNorm(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Token type embeddings
        self.token_type_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.token_type_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, point_patches, clip_image, t):
        B = point_patches.size(0)
        device = point_patches.device

        # === Extract Features ===
        clip_features = self.clip_att(clip_image)           # (B, 768)
        point_embeddings = self.point_mae(point_patches)    # (B, num_patches, 768)
        time_embeddings = self.time_embed(t)                # (B, 768)

        # === Tokens ===
        image_token = clip_features.unsqueeze(1)
        time_token = time_embeddings.unsqueeze(1)
        fused = torch.cat([point_embeddings, image_token, time_token], dim=1)  # (B, seq_len, 768)

        # === Add token type embeddings ===
        token_types = torch.zeros_like(fused)
        token_types[:, :self.num_patches] = self.token_type_embed[0]
        token_types[:, self.num_patches] = self.token_type_embed[1]
        token_types[:, self.num_patches + 1] = self.token_type_embed[2]
        fused = fused + token_types

        # === Add positional embedding ===
        pos_embed = self.pos_embed(self.seq_length, device)
        fused = fused + pos_embed

        fused = self.input_norm(fused)
        fused = self.dropout(fused)

        # === Transformer blocks ===
        for blk in self.blocks:
            fused = blk(fused)

        fused = self.final_norm(fused)
        return fused

    @staticmethod
    def get_model_size(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        return total_params, trainable_params, model_size_mb


# ------------------------------------------------------------------
# Example Run
# ------------------------------------------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiffPoint_ViT_Encoder().to(device)
    _,_,size = model.get_model_size(model)
    print(size)
    # Dummy input   
    B = 2
    point_patches = torch.randn(B, 32, 32, 3).to(device)
    clip_image = torch.randn(B, 5, 768).to(device)
    t = torch.randint(0, 1000, (B,)).float().to(device)

    # Forward pass
    out = model(point_patches, clip_image, t)
    print("Output:", out.shape)
