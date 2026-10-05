# train_diffusion.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from diffusion_model import DiffusionPointModel, loss_mse_noise, loss_chamfer
from diffusion_model import chamfer_distance  # optional
# Import or create your embedding / vit classes
from embedding import Projection_Secure_EmbeddingLayer as EmbeddingModule
from transformer import DiffusionViT as DiffusionViTBackbone

# -------------------------
# Simple FPS + KNN helpers (copy/adapted from your earlier helpers)
# -------------------------
def fps_batch(batch_points: torch.Tensor, n_samples: int):
    # batch_points: (B, N, 3)
    device = batch_points.device
    B, N, _ = batch_points.shape
    centroids = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distances = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), device=device)
    batch_idx = torch.arange(B, device=device)
    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid_xyz = batch_points[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.sum((batch_points - centroid_xyz) ** 2, dim=-1)
        distances = torch.minimum(distances, dist)
        farthest = torch.max(distances, dim=1)[1]
    return centroids  # (B, n_samples)


def knn_grouping(xyz: torch.Tensor, centroids_idx: torch.Tensor, k: int):
    """
    xyz: (B, N, 3)
    centroids_idx: (B, S)
    returns: patches: (B, S, k, 3)
    """
    B, N, _ = xyz.shape
    S = centroids_idx.shape[1]
    centroids_xyz = xyz[torch.arange(B)[:, None], centroids_idx]  # (B, S, 3)
    dist = torch.cdist(centroids_xyz, xyz)  # (B, S, N)
    idx = dist.topk(k, dim=-1, largest=False)[1]  # (B, S, k)
    # gather
    # expand xyz: (B, 1, N, 3) then use idx expanded to (B, S, k, 3)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    xyz_exp = xyz.unsqueeze(1).expand(-1, S, -1, -1)
    patches = torch.gather(xyz_exp, 2, idx_exp)  # (B, S, k, 3)
    return patches


# -------------------------
# Dataset for your format
# -------------------------
class PointCloudFolderDataset(Dataset):
    def __init__(self, root_dir, batches=('Batch_1','Batch_2','Batch_3'), num_patches=64, points_per_patch=16, transform=None):
        r"""
        root_dir: base dataset dir (e.g. H:\Dataset)
        each batch folder contains many model folders. Each model folder has clip_features.npy and points.npy
        """
        self.root_dir = root_dir
        self.batch_dirs = [os.path.join(root_dir, b) for b in batches]
        self.entries = []
        for bd in self.batch_dirs:
            if not os.path.isdir(bd):
                continue
            for model_folder in sorted(os.listdir(bd)):
                fp = os.path.join(bd, model_folder)
                if not os.path.isdir(fp): 
                    continue
                clip_p = os.path.join(fp, 'clip_features.npy')
                pts_p = os.path.join(fp, 'points.npy')
                if os.path.exists(clip_p) and os.path.exists(pts_p):
                    self.entries.append((clip_p, pts_p))
        self.num_patches = num_patches
        self.points_per_patch = points_per_patch
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        clip_path, pts_path = self.entries[idx]
        clip = np.load(clip_path)  # shape depends, e.g., (6, 768) or (num_views, 768)
        pts = np.load(pts_path)    # (N, 3)
        # convert to torch
        clip_t = torch.from_numpy(clip).float()
        pts_t = torch.from_numpy(pts).float()
        # ensure some min points
        if pts_t.ndim == 2 and pts_t.shape[1] == 3:
            pass
        else:
            raise ValueError("points.npy must be shape (N,3)")

        return clip_t, pts_t


# -------------------------
# collate & patchify utility
# -------------------------
def collate_and_patchify(batch, num_patches=64, points_per_patch=16, device='cpu'):
    """
    batch: list of tuples (clip_t, pts_t)
    returns:
      clip_batch: (B, num_views, feat_dim) --- here we won't modify clip shape
      point_patches: (B, num_patches, points_per_patch, 3)
      coords_centers: (B, num_patches, 3) center coords for positional MLP
      x0_points: (B, Npoints_total, 3)
    """
    clip_list = []
    pts_list = []
    for clip_t, pts_t in batch:
        clip_list.append(clip_t)
        pts_list.append(pts_t)

    B = len(clip_list)
    # pad/stack clip features: if clips have variable shape, make them (B, V, D)
    clip_batch = torch.stack([c if c.ndim==2 else c.unsqueeze(0) for c in clip_list]).to(device)

    # build point cloud batch (pad to same N if needed)
    maxN = max([p.shape[0] for p in pts_list])
    pts_padded = torch.zeros(B, maxN, 3, device=device)
    valid_N = []
    for i, p in enumerate(pts_list):
        n = p.shape[0]
        pts_padded[i, :n, :] = p.to(device)
        valid_N.append(n)

    # For each sample compute centroids (FPS) and extract patches via KNN
    # We will use first num_patches centers; if point count < required, sample with replacement
    point_patches = torch.zeros(B, num_patches, points_per_patch, 3, device=device)
    centers = torch.zeros(B, num_patches, 3, device=device)
    x0_points = []  # final flattened target clouds (we may sample or pad to fixed size)
    for i in range(B):
        xyz = pts_padded[i:i+1, :valid_N[i], :]  # (1, Ni, 3)
        Ni = valid_N[i]
        if Ni < 2:
            raise ValueError("point cloud has fewer than 2 points")
        # fps on this point set
        cent_idx = fps_batch(xyz, n_samples=num_patches)[0]  # (num_patches,)
        cent_idx = cent_idx.to(device)
        centers[i] = xyz[0, cent_idx, :]
        # knn
        patches = knn_grouping(xyz, cent_idx.unsqueeze(0), k=points_per_patch)  # (1, num_patches, k, 3)
        point_patches[i] = patches[0]
        # x0 points: we use the original point cloud, optionally sample/pad to num_patches*points_per_patch
        desired = num_patches * points_per_patch
        pts = pts_padded[i, :valid_N[i], :]
        if valid_N[i] >= desired:
            sampled_idx = torch.randperm(valid_N[i], device=device)[:desired]
            x0_flat = pts[sampled_idx]
        else:
            # pad by repeating
            repeats = (desired + valid_N[i] - 1) // valid_N[i]
            cat = pts.repeat(repeats, 1)[:desired]
            x0_flat = cat
        x0_points.append(x0_flat)
    x0_batch = torch.stack(x0_points, dim=0)  # (B, desired, 3)

    return clip_batch, point_patches, centers, x0_batch


# -------------------------
# Trainer
# -------------------------
def train(
    dataset_root,
    out_dir,
    epochs=100,
    batch_size=8,
    lr=1e-4,
    device=None,
    num_patches=64,
    points_per_patch=16,
    timesteps=1000,
    use_chamfer=False
):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # Build embedding module and ViT backbone
    embed_module = EmbeddingModule(embed_dim=768, num_patches=num_patches)
    # You must supply your ViT backbone class. Example placeholder:
    vit_backbone = DiffusionViTBackbone(embed_dim=768, depth=19, num_patches=num_patches, patch_size=points_per_patch)

    model = DiffusionPointModel(vit_encoder=vit_backbone, embed_module=embed_module,
                                embed_dim=768, num_patches=num_patches, points_per_patch=points_per_patch, timesteps=timesteps, device=device)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # dataset & dataloader
    ds = PointCloudFolderDataset(root_dir=dataset_root, batches=('Batch_1','Batch_2','Batch_3'),
                                 num_patches=num_patches, points_per_patch=points_per_patch)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_and_patchify(x, num_patches=num_patches, points_per_patch=points_per_patch, device=device), num_workers=2)

    os.makedirs(out_dir, exist_ok=True)
    global_step = 0

    for epoch in range(epochs):
        model.train()
        for (clip_batch, point_patches, centers, x0_batch) in dl:
            B = x0_batch.shape[0]
            t = torch.randint(0, timesteps, (B,), device=device, dtype=torch.long)
            # forward
            out = model(x0_batch, t, clip_image=clip_batch, point_patches_coords=centers)
            pred = out['pred_points']  # (B, predicted_points, 3)
            # if model predicted shape might be different; ensure equal dims
            if pred.shape[1] != x0_batch.shape[1]:
                # If pred shape differs, try to pad or trim
                desired = x0_batch.shape[1]
                if pred.shape[1] > desired:
                    pred = pred[:, :desired, :]
                else:
                    # pad by repeating
                    repeats = (desired + pred.shape[1] - 1) // pred.shape[1]
                    pred = pred.repeat(1, repeats, 1)[:, :desired, :]

            if use_chamfer:
                loss = loss_chamfer(pred, x0_batch)
            else:
                # predict noise: invert q_sample: if model outputs x0_pred, compute implied eps
                # But above we predicted points from tokens given xt; to use Lsimple (noise), you'd need the model to predict noise directly.
                # Simpler: use MSE between pred and x0:
                loss = F.mse_loss(pred, x0_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            global_step += 1
            if global_step % 10 == 0:
                print(f"Epoch {epoch} Step {global_step} Loss {loss.item():.6f}")

            if global_step % 500 == 0:
                ckpt = os.path.join(out_dir, f"checkpoint_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict()
                }, ckpt)
                print("Saved", ckpt)

        scheduler.step()

    # final save
    torch.save({"model_state": model.state_dict()}, os.path.join(out_dir, "final_model.pt"))
    print("Training finished, model saved.")

# -------------------------
# If called directly, run a small smoke train
# -------------------------
if __name__ == "__main__":
    dataset_root = r"H:\Dataset"  # change to your dataset path
    out_dir = "./checkpoints"
    train(dataset_root=dataset_root, out_dir=out_dir, epochs=1, batch_size=2, lr=2e-4,
          num_patches=32, points_per_patch=32, timesteps=200, use_chamfer=True)
