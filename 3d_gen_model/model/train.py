import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from diffusion_model import DiffPoint_DDPM
from vit_encoder import DiffPoint_ViT_Encoder
from sklearn.neighbors import NearestNeighbors
# ================================
# CONFIGURATION
# ================================
DATASET_ROOT = r"F:\Dataset"
BATCH_SIZE = 32
EPOCHS = 350
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
PLOT_SAVE = "./training_progress.png"

RESUME_TRAINING = True
LOAD_MODEL_ONLY = False
PRETRAINED_MODEL_PATH = r'E:\myProgramme\3d_gen_model\model\checkpoints\final_model_20251114_1740.pt'

GRADIENT_CLIP_MAX_NORM = 1.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
LAMBDA_CHAMFER = 1.0  # Weight for Chamfer loss
LAMBDA_MSE = 1.0  # Weight for MSE loss (added back)
NUM_WORKERS = 0

# ================================
# FARTHEST POINT SAMPLING
# ================================
def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling (FPS) for better geometric coverage
    
    Args:
        points: (N, 3) torch tensor
        num_samples: target number of points
    
    Returns:
        sampled_points: (num_samples, 3) torch tensor
    """
    N = points.shape[0]
    
    if N <= num_samples:
        # Pad with duplicates if not enough points
        indices = torch.randint(0, N, (num_samples,), device=points.device)
        return points[indices]
    
    # FPS algorithm
    centroids = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    distance = torch.ones(N, device=points.device) * 1e10
    farthest = torch.randint(0, N, (1,), device=points.device).item()
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :].view(1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance).item()
    
    return points[centroids]

def group_knn(points, centers, k):
    """
    Group k-nearest neighbors around each center.
    Args:
        points: (N, 3) tensor
        centers: (num_patches, 3) tensor
        k: number of neighbors
    Returns:
        grouped_points: (num_patches, k, 3)
    """
    points_np = points.cpu().numpy()
    centers_np = centers.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points_np)
    _, indices = nbrs.kneighbors(centers_np)
    grouped = torch.tensor(points_np[indices], dtype=torch.float32, device=points.device)
    return grouped
# ================================
# DATASET
# ================================
class DiffPointDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        
        for batch_name in ["Batch_1", "Batch_2", "Batch_3"]:
            batch_path = os.path.join(root_dir, batch_name)
            if not os.path.isdir(batch_path): 
                continue
            for model_name in os.listdir(batch_path):
                model_path = os.path.join(batch_path, model_name)
                if not os.path.isdir(model_path):
                    continue
                clip_path = os.path.join(model_path, "clip_features.npy")
                # Use cleaned_points.npy (normalized + SOR applied)
                pc_path = os.path.join(model_path, "cleaned_points.npy")
                if not os.path.exists(pc_path):
                    # fallback if cleaned file not found
                    print(f'⚠️ cleaned_points.npy not found for {model_name}, using points.npy')
                    pc_path = os.path.join(model_path, "points.npy")
                if os.path.exists(clip_path) and os.path.exists(pc_path):
                    self.samples.append((clip_path, pc_path))
        
        print(f"📂 Found {len(self.samples)} samples in dataset")
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        clip_path, pc_path = self.samples[idx]
        clip_features = np.load(clip_path)
        point_cloud = np.load(pc_path)
        return clip_features, point_cloud


def collate_fn(batch, verbose=False):
    """
    - FPS selects num_patches center points
    - KNN gathers local neighborhoods (points_per_patch)
    - Output: [B, num_patches, num_points_per_patch, 3]
    NO NORMALIZATION - Data already normalized to [-1, 1]
    """
    NUM_PATCHES = 32
    POINTS_PER_PATCH = 32

    clip_features_list = []
    point_clouds_grouped = []
    
    for clip_feat, pc in batch:
        clip_features = torch.tensor(clip_feat, dtype=torch.float32)
        point_cloud = torch.tensor(pc, dtype=torch.float32)

        # NO NORMALIZATION HERE - data already in [-1, 1]
        
        # Farthest Point Sampling → patch centers
        centers = farthest_point_sampling(point_cloud, NUM_PATCHES)

        # KNN grouping → local neighborhoods
        grouped = group_knn(point_cloud, centers, POINTS_PER_PATCH)  # (32, 32, 3)

        clip_features_list.append(clip_features)
        point_clouds_grouped.append(grouped)

    clip_features_batch = torch.stack(clip_features_list)
    point_clouds_batch = torch.stack(point_clouds_grouped)  # [B, 32, 32, 3]

    if verbose:
        print(f"   📊 Grouped batch shape: {point_clouds_batch.shape}")
        print(f"   📊 Range: [{point_clouds_batch.min():.6f}, {point_clouds_batch.max():.6f}]")

    return point_clouds_batch, clip_features_batch
# ================================
# TRAINING FUNCTION
# ================================
def train_model(model, dataloader, optimizer, scheduler, num_epochs=EPOCHS, start_epoch=0, best_loss=float("inf")):
    """
    FIXED: Proper DDPM training loop
    - Pass only ground truth (clean) point clouds to the model
    - Let the DDPM handle noise addition internally
    - Include both MSE and Chamfer losses
    """
    min_lr = 1e-9  # Set the minimum learning rate

    losses, mse_losses, chamfer_losses, lrs = [], [], [], []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss, epoch_mse, epoch_chamfer = 0, 0, 0

        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (gt_pc, clip_features) in loop:
            gt_pc, clip_features = gt_pc.to(DEVICE), clip_features.to(DEVICE)

            optimizer.zero_grad()
            
            # Sample random timesteps for each sample in batch
            t = torch.randint(0, model.timesteps, (gt_pc.shape[0],), device=DEVICE).long()

            # Compute the loss (including both MSE and Chamfer loss)
            loss, mse_loss, chamfer_loss, x_0_pred = model.compute_loss(gt_pc, clip_features, t)
            max_val = x_0_pred.max().item()
            min_val = x_0_pred.min().item()

            if abs(max_val) > 10 or abs(min_val) > 10:
                print(f"⚠️ WARNING: Output exploding! Range = [{min_val:.2f}, {max_val:.2f}]")
            else:
                print(f"Epoch OK: Range = [{min_val:.2f}, {max_val:.2f}]")

            # FIXED: Include both MSE and Chamfer losses
            # MSE loss is fundamental to DDPM (minimizes noise prediction error)
            # Chamfer loss ensures geometric quality
            weighted_loss = LAMBDA_MSE * mse_loss + LAMBDA_CHAMFER * chamfer_loss

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)   # L2 norm of gradients
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            print(f"Gradient L2 norm: {total_norm:.6f}")
            optimizer.step()

            # Check and enforce the minimum learning rate
            if optimizer.param_groups[0]['lr'] < min_lr:
                optimizer.param_groups[0]['lr'] = min_lr
                print(f"Learning rate reached the minimum threshold: {min_lr}")

            epoch_loss += weighted_loss.item()
            epoch_mse += mse_loss.item()
            epoch_chamfer += chamfer_loss.item()

            loop.set_postfix({
                "Loss": weighted_loss.item(), 
                "MSE": mse_loss.item(),
                "Chamfer": chamfer_loss.item()
            })

        avg_loss = epoch_loss / len(dataloader)
        avg_mse = epoch_mse / len(dataloader)
        avg_chamfer = epoch_chamfer / len(dataloader)

        losses.append(avg_loss)
        mse_losses.append(avg_mse)
        chamfer_losses.append(avg_chamfer)
        lrs.append(optimizer.param_groups[0]['lr'])

        scheduler.step(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(SAVE_DIR, f"best_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, best_path)
            print(f"✅ Best model saved: {best_path}")

        # Plot training curves
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 4, 1)
        plt.plot(losses, label="Total Loss", color='blue')
        plt.title("Total Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(True); plt.legend()

        plt.subplot(1, 4, 2)
        plt.plot(mse_losses, label="MSE Loss", color='orange')
        plt.title("MSE Loss (Noise Prediction)"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(True); plt.legend()

        plt.subplot(1, 4, 3)
        plt.plot(chamfer_losses, label="Chamfer Loss", color='green')
        plt.title("Chamfer Loss (Geometry)"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(True); plt.legend()

        plt.subplot(1, 4, 4)
        plt.plot(lrs, label="LR", color='red')
        plt.title("Learning Rate"); plt.xlabel("Epoch"); plt.ylabel("LR")
        plt.yscale('log'); plt.grid(True); plt.legend()

        plt.tight_layout()
        plt.savefig(PLOT_SAVE, dpi=150)
        plt.close()

        print(f"Epoch [{epoch+1}/{num_epochs}] | Total: {avg_loss:.6f} | MSE: {avg_mse:.6f} | Chamfer: {avg_chamfer:.6f} | LR: {lrs[-1]:.6e}")

    return losses, mse_losses, chamfer_losses

# ================================
# MAIN ENTRY POINT
# ================================


if __name__ == "__main__":
    print("🚀 Starting DiffPoint-DDPM Training (FIXED VERSION)")
    print("=" * 70)
    print("✅ FIXES APPLIED:")
    print("   1. Using Farthest Point Sampling instead of random sampling")
    print("   2. Keeping point clouds in standard (N, 3) format")
    print("   3. Removed incorrect noisy_pc generation from collate_fn")
    print("   4. DDPM handles noise addition internally")
    print("   5. Including both MSE and Chamfer losses")
    print("=" * 70)

    dataset = DiffPointDataset(DATASET_ROOT)
    if len(dataset) == 0:
        raise RuntimeError("❌ No valid (clip_features.npy, cleaned_points.npy) pairs found.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        collate_fn=lambda batch: collate_fn(batch)
    )

    # Initialize model
    vit_encoder = DiffPoint_ViT_Encoder(embed_dim=768)
    model = DiffPoint_DDPM(
        vit_encoder=vit_encoder, 
        embed_dim=768, 
        num_patches=32, 
        num_points_per_patch=32
    )
    model.to(DEVICE)
    print(f"📍 Model device: {next(model.parameters()).device}")

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        factor=SCHEDULER_FACTOR, 
        patience=SCHEDULER_PATIENCE, 
        verbose=True
    )

    # Resume training if checkpoint exists
    start_epoch = 0
    best_loss = float("inf")
    
    if RESUME_TRAINING and PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"📥 Loading checkpoint from: {PRETRAINED_MODEL_PATH}")
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not LOAD_MODEL_ONLY:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', float("inf"))
        print(f"✅ Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}")
        new_learning_rate = 1e-4 # Set your new learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_learning_rate
        print(f"✅ Learning rate reset to: {new_learning_rate}")

    print("\n🏋️ Starting training loop...")
    losses, mse_losses, chamfer_losses = train_model(
        model, 
        dataloader, 
        optimizer, 
        scheduler, 
        num_epochs=EPOCHS,
        start_epoch=start_epoch,
        best_loss=best_loss
    )

    # Save final model
    final_path = os.path.join(SAVE_DIR, f"final_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pt")
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
        'mse_losses': mse_losses,
        'chamfer_losses': chamfer_losses,
        'best_loss': min(losses) if losses else float("inf")
    }, final_path)
    if(len(losses)>0):
        print("=" * 70)
        print(f"🎯 Training complete! Final model saved at: {final_path}")
        print(f"📊 Final metrics:")
        print(f"   - Total Loss: {losses[-1]:.6f}")
        print(f"   - MSE Loss: {mse_losses[-1]:.6f}")
        print(f"   - Chamfer Loss: {chamfer_losses[-1]:.6f}")
        print(f"   - Best Loss: {min(losses):.6f}")
        print("=" * 70)