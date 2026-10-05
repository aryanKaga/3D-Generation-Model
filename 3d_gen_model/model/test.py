import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
from diffusion_model import DiffPoint_DDPM
from vit_encoder import DiffPoint_ViT_Encoder

# ===================================================
# CONFIGURATION
# ===================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"E:\myProgramme\3d_gen_model\model\checkpoints\best_model_epoch_1740.pt"  # ✅ update if needed
CLIP_FEATURE_PATH = r"F:\Dataset\Batch_3\5c7a1fdf9c574fa4b20aca2029fd1ca4\clip_features.npy"  # ✅ update if needed

# Sampling settings
NUM_PATCHES = 32
NUM_POINTS_PER_PATCH = 32
SAMPLE_STEPS = 50 # should match your training diffusion timesteps

SAVE_OUTPUT = True
OUTPUT_PATH = "./generated_pointcloud.npy"
SHOW_VISUALIZATION = True


# ===================================================
# UTILITY FUNCTIONS
# ===================================================
def visualize_point_cloud(pc, title="Generated Point Cloud", color="blue"):
    """Render a 3D point cloud."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=color, s=3, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # keep aspect ratio consistent
    plt.tight_layout()
    plt.show()


# ===================================================
# MAIN INFERENCE FUNCTION
# ===================================================
@torch.no_grad()
def generate_point_cloud(model, clip_features):
    """
    Use the trained DDPM to generate a clean 3D point cloud
    conditioned on CLIP features.
    """
    model.eval()

    # ✅ ensure tensor + correct shape
    if isinstance(clip_features, np.ndarray):
        clip_features = torch.tensor(clip_features, dtype=torch.float32)
    clip_features = clip_features.to(DEVICE).unsqueeze(0)

    B = 1  # single sample

    # ✅ initialize random Gaussian noise
    x = (torch.rand((B, NUM_PATCHES, NUM_POINTS_PER_PATCH, 3), device=DEVICE) - 0.5)

    print(torch.mean(x),torch.max(x),torch.min(x),torch.std(x))
    print("🚀 Starting denoising diffusion sampling...")

    for t in reversed(range(model.timesteps)):
        t_tensor = torch.full((B,), t, device=DEVICE, dtype=torch.long)
        # ✅ use p_sample (the correct method in DDPM)
        x = model.p_sample(x, clip_features, t_tensor)

        # Print status periodically
        if t % 100 == 0 or t < 10:
            print(f"   Step {t}/{model.timesteps} done.")

    # ✅ reshape to a single Nx3 array
    generated_pc = x.squeeze(0).reshape(-1, 3).detach().cpu().numpy()

    # ✅ range diagnostics
    print("\n===== DIAGNOSTICS =====")
    print(f"Generated PC → shape: {generated_pc.shape}")
    print(f"X range: {generated_pc[:,0].min():.4f} → {generated_pc[:,0].max():.4f}")
    print(f"Y range: {generated_pc[:,1].min():.4f} → {generated_pc[:,1].max():.4f}")
    print(f"Z range: {generated_pc[:,2].min():.4f} → {generated_pc[:,2].max():.4f}")
    print(f"CLIP features range: {clip_features.min().item():.4f} → {clip_features.max().item():.4f}")
    print("========================\n")

    return generated_pc


# ===================================================
# MAIN EXECUTION
# ===================================================
if __name__ == "__main__":
    print("🚀 Loading pretrained DiffPoint-DDPM model...")

    # === Load model ===
    vit_encoder = DiffPoint_ViT_Encoder(embed_dim=768)
    model = DiffPoint_DDPM(
        vit_encoder=vit_encoder,
        embed_dim=768,
        num_patches=NUM_PATCHES,
        num_points_per_patch=NUM_POINTS_PER_PATCH,
    ).to(DEVICE)

    # === Load checkpoint ===
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Loaded model from {MODEL_PATH}\n")

    # === Load CLIP features ===
    if not os.path.exists(CLIP_FEATURE_PATH):
        raise FileNotFoundError(f"❌ clip_features.npy not found: {CLIP_FEATURE_PATH}")

    clip_features = np.load(CLIP_FEATURE_PATH)
    print(f"📦 Loaded CLIP features: {clip_features.shape}")

    # === Generate point cloud ===
    generated_pc = generate_point_cloud(model, clip_features)

    print(f"✅ Generated point cloud shape: {generated_pc.shape}")

    # === Save + visualize ===
    if SAVE_OUTPUT:
        np.save(OUTPUT_PATH, generated_pc)
        print(f"💾 Saved generated point cloud to: {OUTPUT_PATH}")

    if SHOW_VISUALIZATION:
        visualize_point_cloud(generated_pc, "Generated 3D Model (DDPM Output)")
