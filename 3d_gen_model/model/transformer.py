import torch
import torch.nn as nn

# --------------------------------------------------
# MLP Positional Encoding (following DiffPoint)
# --------------------------------------------------
class MLPPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, coords):
        """
        coords: [B, N, 3] - raw xyz coordinates or noisy point cloud
        returns: [B, N, embed_dim]
        """
        return self.mlp(coords)


# --------------------------------------------------
# Transformer Block (standard ViT)
# --------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------
# Diffusion ViT Backbone with 3D projection
# --------------------------------------------------
class DiffusionViT(nn.Module):
    def __init__(self, embed_dim=768, depth=19, num_heads=12, mlp_ratio=4.0, dropout=0.0,
                 num_patches=64, patch_size=16):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_size = patch_size

        # MLP positional embedding
        self.pos_mlp = MLPPositionalEncoding(embed_dim=embed_dim)

        # Learnable positional encoding for tokens (point + image + time)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer backbone (19 layers)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Final projection head (MLP → 3D)
        self.fc_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3)
        )

    def forward(self, token_embeddings, coords):
        """
        token_embeddings: [B, num_patches+2, embed_dim]
        coords: [B, num_patches, 3]  (for positional encoding via MLP)
        Returns:
            predicted_points: [B, 1024, 3]
        """
        B, S, D = token_embeddings.shape

        # Generate positional embeddings using MLP on coordinates
        pos_embed = self.pos_mlp(coords)  # [B, num_patches, embed_dim]

        # Concatenate learnable embeddings (+ image/time tokens)
        x = token_embeddings.clone()
        x[:, :self.num_patches, :] += pos_embed
        x = x + self.pos_embed[:, :S, :]

        # Pass through Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Take only point tokens for reconstruction
        point_tokens = x[:, :self.num_patches, :]  # [B, num_patches, embed_dim]

        # Predict 3D coordinates via MLP projection
        predicted_patches = self.fc_proj(point_tokens)  # [B, num_patches, 3]

        # Upsample or reshape to full 1024-point cloud (64 patches × 16 pts each)
        predicted_points = predicted_patches.repeat_interleave(self.patch_size, dim=1)
        predicted_points = predicted_points[:, :1024, :]  # Ensure fixed size

        return predicted_points
