import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------------
# FPS: Farthest Point Sampling
# -------------------------
def fps_batch(batch, n_samples=32):
    B, N, _ = batch.shape
    centroids = torch.zeros(B, n_samples, dtype=torch.long, device=batch.device)
    distances = torch.ones(B, N, device=batch.device) * 1e10
    farthest = torch.randint(0, N, (B,), device=batch.device)
    batch_indices = torch.arange(B, device=batch.device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = batch[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((batch - centroid) ** 2, dim=-1)
        distances = torch.minimum(distances, dist)
        farthest = torch.max(distances, dim=1)[1]
    return centroids


# -------------------------
# KNN for grouping neighbors
# -------------------------
def knn_points(xyz, centroids, k):
    B, N, _ = xyz.shape
    s = centroids.shape[1]
    centroid_pts = xyz[torch.arange(B)[:, None], centroids]  # (B, s, 3)
    dist = torch.cdist(centroid_pts, xyz)  # (B, s, N)
    idx = dist.topk(k, dim=-1, largest=False)[1]  # nearest k points
    patches = xyz[:, None, :, :].expand(-1, s, -1, -1).gather(2, idx[..., None].expand(-1, -1, -1, 3))
    return patches  # (B, s, k, 3)




a = torch.randn(1,32,32)
inpt = torch.randn(1,1024,3)
centroids = fps_batch(inpt)

otpt = knn_points(inpt,centroids,32)
print(otpt.shape)