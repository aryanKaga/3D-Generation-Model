import os
import numpy as np
import open3d as o3d

# ==========================
# Helper Functions
# ==========================
def normalize_pc(points):
    """Normalize point cloud to fit inside a unit sphere."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))
    points = points / furthest_distance
    return points

def perform_sor(points, nb_neighbors=20, std_ratio=2.0):
    """Perform Statistical Outlier Removal (SOR)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(pcd.points)

# ==========================
# Folder Processing
# ==========================
def process_folder(folder_path):
    """
    For each folder:
    1. Load point_clouds.npy
    2. Normalize
    3. Perform SOR
    4. Save as cleaned_points.npy
    """
    pointcloud_path = os.path.join(folder_path, "points.npy")
    cleaned_path = os.path.join(folder_path, "cleaned_points.npy")

    if not os.path.exists(pointcloud_path):
        print(f"[SKIP] point_clouds.npy not found in {folder_path}")
        return

    try:
        # Step 1: Load
        points = np.load(pointcloud_path)

        # Step 2: Normalize
        normalized_points = normalize_pc(points)

        # Step 3: Perform SOR
        print(f"[INFO] Normalizing + Performing SOR on {folder_path}")
        cleaned_points = perform_sor(normalized_points)

        # Step 4: Save
        np.save(cleaned_path, cleaned_points)
        print(f"[DONE] Saved cleaned_points.npy in {folder_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process {folder_path}: {e}")

# ==========================
# Main Execution
# ==========================
def process_dataset(root_dir):
    """
    Traverse structure:
    Dataset/
        Batch_x/
            folder/
                point_clouds.npy
    """
    for batch in os.listdir(root_dir):
        batch_path = os.path.join(root_dir, batch)
        if not os.path.isdir(batch_path):
            continue

        for folder in os.listdir(batch_path):
            folder_path = os.path.join(batch_path, folder)
            if os.path.isdir(folder_path):
                process_folder(folder_path)

if __name__ == "__main__":
    ROOT = r"F:\Dataset"  # Root directory containing Batch_1, Batch_2, etc.
    process_dataset(ROOT)
