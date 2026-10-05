import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================================================================================
# DDPM Noise Schedule (Cosine for High Quality)
# ==================================================================================
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule - better quality than linear for point clouds.
    From "Improved Denoising Diffusion Probabilistic Models"
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DDPMScheduler:
    """Manages DDPM noise schedule"""
    def __init__(self, timesteps=1000, beta_schedule='cosine'):
        self.timesteps = timesteps
        
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer = lambda name, val: setattr(self, name, val)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance', 
                           betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))


# ==================================================================================
# Enhanced Chamfer Distance with Batch Optimization
# ==================================================================================


def chamfer_distance(pred, target, reduction='mean'):
    """
    L1 Chamfer Distance for point clouds.

    Args:
        pred: (B, N, 3) - Predicted points (batch_size, num_points, 3)
        target: (B, M, 3) - Target points (batch_size, num_points, 3)
        reduction: 'mean' or 'sum' - Method of reducing the result (default: 'mean')
    
    Returns:
        Chamfer distance loss (scalar)
    """
    # Step 1: Expand dimensions to allow broadcasting
    # pred: (B, N, 3) -> pred_expanded: (B, N, 1, 3)
    # target: (B, M, 3) -> target_expanded: (B, 1, M, 3)
    pred_expanded = pred.unsqueeze(2)      # Shape becomes (B, N, 1, 3)
    target_expanded = target.unsqueeze(1)  # Shape becomes (B, 1, M, 3)

    # Step 2: Compute pairwise L1 distances (absolute differences)
    # dist: (B, N, M) - L1 distance between every pair of points in pred and target
    dist = torch.sum(torch.abs(pred_expanded - target_expanded), dim=-1)  # (B, N, M)

    # Step 3: Compute the nearest neighbor distances
    # min_dist_pred_to_target: (B, N) - Minimum distance for each pred point to the nearest target point
    min_dist_pred_to_target, _ = torch.min(dist, dim=2)  # (B, N)

    # min_dist_target_to_pred: (B, M) - Minimum distance for each target point to the nearest pred point
    min_dist_target_to_pred, _ = torch.min(dist, dim=1)  # (B, M)

    # Step 4: Compute the Chamfer Distance (symmetric)
    if reduction == 'mean':
        # Average the distances over all points
        cd = torch.mean(min_dist_pred_to_target) + torch.mean(min_dist_target_to_pred)
    elif reduction == 'sum':
        # Sum the distances over all points
        cd = torch.sum(min_dist_pred_to_target) + torch.sum(min_dist_target_to_pred)
    else:
        # Optionally return the mean distance per sample in the batch
        cd = min_dist_pred_to_target.mean(dim=1) + min_dist_target_to_pred.mean(dim=1)  # (B,)

    return cd


# ==================================================================================
# High-Accuracy Point Decoder with Skip Connections + TANH OUTPUT
# ==================================================================================
class PointDecoder(nn.Module):
    """
    Enhanced decoder with skip connections, layer normalization, and TANH output.
    Progressively upsamples embeddings to point coordinates.
    """
    def __init__(self, embed_dim=768, num_points_per_patch=32, output_dim=3, hidden_dims=[512, 384, 256]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_points = num_points_per_patch
        self.output_dim = output_dim
        
        # Progressive decoding with residuals
        layers = []
        in_dim = embed_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # Final projection to point coordinates
        layers.append(nn.Linear(in_dim, num_points_per_patch * output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (B, num_patches, embed_dim)
        Returns:
            (B, num_patches, num_points, output_dim) - BOUNDED to [-1, 1] via tanh
        """
        B, N, D = x.shape
        
        # Decode
        x = self.decoder(x)  # (B, N, num_points * output_dim)
        
        # Reshape to point cloud
        x = x.view(B, N, self.num_points, self.output_dim)
        
        # ✅ CRITICAL FIX: Apply tanh to bound output to [-1, 1]
        x = torch.tanh(x)
        
        return x


# ==================================================================================
# DiffPoint DDPM - FIXED: Direct X₀ Prediction with Tanh
# ==================================================================================
class DiffPoint_DDPM(nn.Module):
    """
    FIXED High-Accuracy DDPM for Point Cloud Generation.
    
    KEY CHANGES:
    ✅ Predicts X₀ directly (not noise ε)
    ✅ Uses torch.tanh() to bound output to [-1, 1]
    ✅ No division by small √ᾱₜ (prevents numerical explosion)
    ✅ Pure Chamfer Distance loss (no MSE on noise)
    
    Args:
        vit_encoder: Pre-built ViT encoder
        num_patches: Number of patches (32)
        num_points_per_patch: Points per patch (32)
        embed_dim: Embedding dimension (768)
        timesteps: Diffusion steps (1000)
        beta_schedule: 'cosine' (recommended) or 'linear'
    """
    def __init__(self, vit_encoder, num_patches=32, num_points_per_patch=32,
                 embed_dim=768, timesteps=1000, beta_schedule='cosine'):
        super().__init__()
        
        self.num_patches = num_patches
        self.num_points = num_points_per_patch
        self.embed_dim = embed_dim
        self.timesteps = timesteps
        
        # ViT Encoder (handles all conditioning)
        self.encoder = vit_encoder
        
        # Enhanced decoder with TANH output
        self.decoder = PointDecoder(
            embed_dim=embed_dim,
            num_points_per_patch=num_points_per_patch,
            output_dim=3,
            hidden_dims=[512, 384, 256]
        )
        
        # DDPM scheduler
        self.scheduler = DDPMScheduler(timesteps, beta_schedule)
        self._register_scheduler_buffers()
        
        # Loss weights
        self.chamfer_weight = 1.0
    
    def _register_scheduler_buffers(self):
        """Register scheduler buffers to this model"""
        for name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                     'posterior_variance', 'posterior_mean_coef1', 'posterior_mean_coef2']:
            self.register_buffer(name, getattr(self.scheduler, name))
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0)
        Add noise to clean point cloud
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_x0(self, x_t, clip_image, t):
        """
        ✅ FIXED: Directly predict clean X₀ (not noise)
        
        Args:
            x_t: (B, num_patches, num_points, 3) - Noisy input
            clip_image: (B, 6, 768) - CLIP conditioning
            t: (B,) - Timestep
        Returns:
            (B, num_patches, num_points, 3) - Predicted CLEAN point cloud (bounded [-1,1])
        """
        # Encode with ViT (handles 3D points directly)
        embeddings = self.encoder(x_t, clip_image, t)  # (B, 34, 768)
        
        # Extract patch embeddings (exclude image and time tokens)
        patch_embeddings = embeddings[:, :self.num_patches, :]  # (B, 32, 768)
        
        # ✅ Decode to predict X₀ directly (with tanh bounding inside decoder)
        x_0_pred = self.decoder(patch_embeddings)  # (B, 32, 32, 3) in [-1, 1]
        
        return x_0_pred
    
    def compute_loss(self, x_start, clip_image, t):
        """
        ✅ FIXED: Pure Chamfer Distance loss on direct X₀ prediction.
        
        No more:
        - Noise prediction
        - Division by √ᾱₜ
        - MSE loss on noise
        
        Args:
            x_start: (B, num_patches, num_points, 3) - Ground truth in [-1, 1]
            clip_image: (B, 6, 768) - CLIP features
            t: (B,) - Timesteps
        Returns:
            total_loss, mse_loss (dummy), chamfer_loss
        """
        B = x_start.shape[0]
        
        # Generate and add noise to create x_t
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        
        # ✅ Directly predict clean X₀ (bounded to [-1,1] via tanh)
        x_0_pred = self.predict_x0(x_t, clip_image, t)
        
        # Reshape: (B, 32, 32, 3) → (B, 1024, 3)
        x_start_flat = x_start.reshape(B, -1, 3)
        x_0_pred_flat = x_0_pred.reshape(B, -1, 3)
        
        # ✅ Pure Chamfer Distance (geometric structure)
        chamfer_loss = chamfer_distance(x_0_pred_flat, x_start_flat)
        
        # Total loss
        total_loss = self.chamfer_weight * chamfer_loss
        
        # Dummy MSE for logging compatibility (not used in backprop)
        mse_loss = torch.tensor(0.0, device=x_start.device)
        
        return total_loss, mse_loss, chamfer_loss,x_0_pred
    
    def forward(self, x_start, clip_image, t):
        """Training forward pass"""
        total_loss, mse_loss, chamfer_loss = self.compute_loss(x_start, clip_image, t)
        return total_loss
    
    @torch.no_grad()
    def p_sample(self, x_t, clip_image, t):
        """
        ✅ FIXED: Single denoising step using direct X₀ prediction.
        
        Uses DDPM posterior formula:
        q(x_{t-1} | x_t, x_0) = N(μ_θ, σ²I)
        where μ_θ = coef1 * x_0_pred + coef2 * x_t
        """
        B = x_t.shape[0]
        
        # ✅ Predict clean X₀ directly (bounded to [-1,1])
        x_0_pred = self.predict_x0(x_t, clip_image, t)
        
        # Clip for numerical stability (though tanh already bounds it)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Get posterior mean coefficients
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        
        # Compute posterior mean
        model_mean = coef1 * x_0_pred + coef2 * x_t
        
        # Add noise (except at t=0)
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, clip_image, batch_size=1, return_all_timesteps=False):
        """
        Generate point cloud from noise (full denoising)
        
        Args:
            clip_image: (B, 6, 768) - CLIP conditioning
            batch_size: Number of samples
            return_all_timesteps: Return intermediate results
        Returns:
            (B, num_patches, num_points, 3) - Generated point cloud in [-1, 1]
        """
        device = clip_image.device
        shape = (batch_size, self.num_patches, self.num_points, 3)
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        if return_all_timesteps:
            all_x = [x]
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, clip_image, t_batch)
            
            if return_all_timesteps:
                all_x.append(x)
        
        if return_all_timesteps:
            return torch.stack(all_x, dim=1)
        return x
    
    @staticmethod
    def get_model_size(model):
        """Calculate model statistics"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        return total_params, trainable_params, model_size_mb


# ==================================================================================
# Example Usage
# ==================================================================================
if __name__ == "__main__":
    from vit_encoder import DiffPoint_ViT_Encoder
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}\n")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {gpu_name} ({device})\n")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (CUDA not available)\n")

    # Create ViT encoder
    vit_encoder = DiffPoint_ViT_Encoder(
        embed_dim=768,
        num_patches=32,
        depth=21,
        num_heads=16,
        mlp_ratio=4.5,
        dropout=0.1,
        drop_path_rate=0.2
    ).to(device)
    
    # Create FIXED high-accuracy DDPM
    diffpoint = DiffPoint_DDPM(
        vit_encoder=vit_encoder,
        num_patches=32,
        num_points_per_patch=32,
        embed_dim=768,
        timesteps=1000,
        beta_schedule='cosine'
    ).to(device)
    
    # Model statistics
    total, trainable, size_mb = DiffPoint_DDPM.get_model_size(diffpoint)
    
    print("=" * 70)
    print("FIXED DIFFPOINT DDPM - Direct X₀ Prediction")
    print("=" * 70)
    print(f"Total parameters:      {total:,}")
    print(f"Trainable parameters:  {trainable:,}")
    print(f"Model size (FP32):     {size_mb:.2f} MB")
    print("=" * 70)
    print("\n✅ KEY FIXES APPLIED:")
    print("   1. Predict X₀ directly (not noise ε)")
    print("   2. Tanh activation → outputs bounded to [-1, 1]")
    print("   3. No division by √ᾱₜ → no numerical explosion")
    print("   4. Pure Chamfer Distance loss")
    print("=" * 70)
    
    # Test forward pass
    batch_size = 2
    point_clouds = torch.randn(batch_size, 32, 32, 3).to(device) * 0.5  # Normalize to ~[-1, 1]
    clip_features = torch.randn(batch_size, 6, 768).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    
    print(f"\nTraining forward pass:")
    print(f"  Point clouds: {point_clouds.shape}, range: [{point_clouds.min():.3f}, {point_clouds.max():.3f}]")
    print(f"  CLIP features: {clip_features.shape}")
    print(f"  Timesteps: {timesteps.shape}")
    
    diffpoint.train()
    total_loss, mse_loss, chamfer_loss = diffpoint.compute_loss(
        point_clouds, clip_features, timesteps
    )
    print(f"\n  Total Loss: {total_loss.item():.6f}")
    print(f"  MSE Loss (dummy): {mse_loss.item():.6f}")
    print(f"  Chamfer Loss: {chamfer_loss.item():.6f}")
    
    # Test sampling
    print(f"\n{'='*70}")
    print("Sampling (Generation)")
    print("=" * 70)
    diffpoint.eval()
    generated = diffpoint.sample(clip_features, batch_size=2)
    print(f"  Generated shape: {generated.shape}")
    print(f"  ✅ Range: [{generated.min().item():.4f}, {generated.max().item():.4f}] (bounded!)")
    
    print("\n" + "="*70)
    print("✅ FIXED High-Accuracy DiffPoint DDPM Ready!")
    print("="*70)
    print("\nKey Features:")
    print("  • Direct X₀ prediction (no noise division)")
    print("  • Tanh output bounding to [-1, 1]")
    print("  • Cosine noise schedule")
    print("  • Pure Chamfer Distance loss")
    print("  • Numerically stable (no explosions!)")
    print("="*70)