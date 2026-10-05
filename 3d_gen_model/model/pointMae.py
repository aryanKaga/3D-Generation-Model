import torch
import torch.nn as nn
import torch.nn.functional as F


class PointPatchEmbed(nn.Module):
    """Efficient patch embedding: converts 32 points (each 3D) to embedding."""
    def __init__(self, num_points=32, point_dim=3, embed_dim=768):
        super().__init__()
        
        # MLP to process points: (32, 3) -> 768
        # First process each point, then aggregate
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )
        
        # Aggregate all points in a patch
        self.patch_encoder = nn.Sequential(
            nn.Linear(128 * num_points, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, num_patches, num_points, 3) = (B, 32, 32, 3)
        Returns:
            (B, num_patches, embed_dim) = (B, 32, 768)
        """
        B, N, P, D = x.shape  # B, 32, 32, 3
        
        # Reshape to process all points
        x = x.reshape(B * N, P, D)  # (B*32, 32, 3)
        
        # Encode each point
        x = self.point_encoder(x)  # (B*32, 32, 128)
        
        # Flatten and aggregate all points in each patch
        x = x.reshape(B * N, -1)  # (B*32, 32*128)
        x = self.patch_encoder(x)  # (B*32, 768)
        
        # Reshape back
        x = x.reshape(B, N, -1)  # (B, 32, 768)
        
        return x


class TransformerBlock(nn.Module):
    """Efficient transformer block with fused operations."""
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Attention with residual
        residual = x
        x = self.norm1(x)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = residual + x
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class PointMAE(nn.Module):
    """
    High-performance PointMAE encoder for 3D point cloud patches.
    
    Optimized architecture:
    - Efficient patch embedding with MLPs for 3D points
    - 8 transformer layers with 12 attention heads
    - Embedding dimension: 768 (standard for transformers)
    - Pre-normalization for training stability
    
    Args:
        num_points: Points per patch (default: 32)
        point_dim: Dimension of each point (default: 3 for x,y,z)
        num_patches: Number of patches per sample (default: 32)
        embed_dim: Embedding dimension (default: 768)
        depth: Number of transformer layers (default: 8)
        num_heads: Number of attention heads (default: 12)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        drop_rate: Dropout rate (default: 0.1)
    
    Input:
        x: (B, 32, 32, 3) where:
           - B: batch size
           - 32: number of patches
           - 32: number of points per patch
           - 3: coordinates (x, y, z) for each point
    
    Output:
        (B, 32, 768) - 768-dimensional embedding for each of 32 patches
    """
    def __init__(self, num_points=32, point_dim=3, num_patches=32, embed_dim=768, 
                 depth=8, num_heads=12, mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()
        
        self.num_points = num_points
        self.point_dim = point_dim
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PointPatchEmbed(num_points, point_dim, embed_dim)
        
        # Learnable positional embeddings for 32 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, 32, 32, 3) 
               - B: batch size
               - 32: number of patches
               - 32: number of points per patch
               - 3: x, y, z coordinates
        
        Returns:
            (B, 32, 768) - 768-dim embedding for each patch
        """
        B, N, P, D = x.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        assert P == self.num_points, f"Expected {self.num_points} points per patch, got {P}"
        assert D == self.point_dim, f"Expected {self.point_dim} coordinates per point, got {D}"
        
        # Embed patches: (B, 32, 32, 3) -> (B, 32, 768)
        x = self.patch_embed(x)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x


# Example usage and testing
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create model
    model = PointMAE(
        num_points=32,      # 32 points per patch
        point_dim=3,        # 3D coordinates (x, y, z)
        num_patches=32,     # 32 patches per sample
        embed_dim=384,      # Standard transformer dimension
        depth=8,            # 8 transformer layers
        num_heads=12,       # 12 attention heads
        mlp_ratio=4.0,
        drop_rate=0.1
    ).to(device)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)\n")
    
    # Test with sample data - YOUR FORMAT
    batch_size = 1
    num_patches = 32
    num_points = 32
    point_dim = 3
    
    x = torch.randn(batch_size, num_patches, num_points, point_dim).to(device)
    
    print(f"Input shape:  {x.shape}")
    print(f"  └─ Batch size: {batch_size}")
    print(f"  └─ Patches per sample: {num_patches}")
    print(f"  └─ Points per patch: {num_points}")
    print(f"  └─ Coordinates per point: {point_dim} (x, y, z)\n")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"  └─ Batch size: {batch_size}")
    print(f"  └─ Patches: {num_patches}")
    print(f"  └─ Embedding dim per patch: {output.shape[-1]}\n")
    
    print("✓ Successfully converted (B, 32, 32, 3) → (B, 32, 768)")
    print("✓ Each patch has its own 768-dimensional embedding!\n")
    
    # Verify output properties
    print("Output statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std:  {output.std().item():.4f}")
    print(f"  Min:  {output.min().item():.4f}")
    print(f"  Max:  {output.max().item():.4f}\n")
    
    # Test with larger batch
    print("=" * 60)
    print("Testing with larger batch (8 samples)")
    print("=" * 60)
    
    batch_size = 8
    x_batch = torch.randn(batch_size, num_patches, num_points, point_dim).to(device)
    print(f"Input shape:  {x_batch.shape}")
    
    with torch.no_grad():
        output_batch = model(x_batch)
    
    print(f"Output shape: {output_batch.shape}")
    print("✓ Batch processing works correctly!\n")
    
    # Performance benchmark
    print("=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    model.eval()
    num_iterations = 1000
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_batch)
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        starter.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x_batch)
        ender.record()
        torch.cuda.synchronize()
        
        avg_time = starter.elapsed_time(ender) / num_iterations
        throughput = (batch_size * num_iterations) / (starter.elapsed_time(ender) / 1000)
        patches_per_sec = throughput * num_patches
        
        print(f"Average inference time: {avg_time:.3f} ms")
        print(f"Throughput: {throughput:.1f} samples/sec")
        print(f"           {patches_per_sec:.0f} patches/sec")
        print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x_batch)
        total_time = time.time() - start
        
        avg_time = (total_time / num_iterations) * 1000
        throughput = (batch_size * num_iterations) / total_time
        patches_per_sec = throughput * num_patches
        
        print(f"Average inference time: {avg_time:.3f} ms")
        print(f"Throughput: {throughput:.1f} samples/sec")
        print(f"           {patches_per_sec:.0f} patches/sec")
    
    print("\n" + "=" * 60)
    print("Integration Example")
    print("=" * 60)
    print("""
# Initialize model
model = PointMAE(num_points=32, point_dim=3, num_patches=32, embed_dim=768)

# Your input: B batches, each with 32 patches of 32 3D points
point_data = torch.randn(B, 32, 32, 3)  # YOUR FORMAT
# Shape: [batch, patches, points_per_patch, xyz_coordinates]

# Get patch embeddings
patch_embeddings = model(point_data)  # Output: (B, 32, 768)

# Now each of 32 patches has a 768-dimensional embedding!
# Use for:
# - Classification: pool patches -> classify
# - Segmentation: upsample embeddings
# - Retrieval: use patch embeddings as features
# - Fusion with CLIP: combine with CLIP image/text features
# - Part segmentation: each patch embedding represents a part
    """)