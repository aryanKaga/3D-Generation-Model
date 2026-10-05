import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim=768, max_period=10000):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # Sinusoidal frequency embedding
        half_dim = embed_dim // 2
        # Fixed: Use proper exponential decay for frequencies
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)
        
        # Learnable MLP projection (common in diffusion models)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
    
    def forward(self, t):
        """
        Args:
            t: Tensor of shape [batch_size] or [batch_size, 1]
               Timestep values (can be normalized [0,1] or integers)
        
        Returns:
            emb: Tensor of shape [batch_size, embed_dim]
        """
        # Ensure t is [batch_size, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() == 2 and t.shape[1] != 1:
            t = t.squeeze()
            if t.dim() == 1:
                t = t.unsqueeze(-1)
        
        # Create sinusoidal embeddings: [batch_size, half_dim]
        args = t.float() * self.freqs.unsqueeze(0)
        
        # Concatenate sin and cos: [batch_size, embed_dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # Handle odd embed_dim (if embed_dim is odd, pad with zero)
        if emb.shape[-1] < self.embed_dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        
        # Project through learnable MLP
        emb = self.mlp(emb)
        
        return emb

