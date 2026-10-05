import clip
import torch
from PIL import Image
import numpy as np
import os
import time

# === CONFIG ===
ROOT_DIR = r"F:\\Dataset"  # change if needed
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
IMAGE_NAMES = ['r_0.png', 'r_1.png', 'r_2.png', 'r_3.png', 'r_4.png', 'r_5.png']

# === DEVICE SETUP ===
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("🚀 Using device: CUDA")
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💽 CUDA version: {torch.version.cuda}")
else:
    print("⚠️ Using CPU only (no CUDA detected)")

# === LOAD CLIP MODEL ===
print("🔄 Loading CLIP ViT-L/14 ...")
model, preprocess = clip.load("ViT-L/14", device=device)
model.eval()
print("✅ Model loaded successfully.")

start_time = time.time()
processed_folders = 0
saved_files = 0

# === PROCESS ALL BATCHES ===
for batch in BATCHES:
    batch_path = os.path.join(ROOT_DIR, batch)
    if not os.path.exists(batch_path):
        print(f"⚠️ Missing batch folder: {batch_path}")
        continue

    print(f"\n=== Processing {batch} ===")

    for model_folder in os.listdir(batch_path):
        model_dir = os.path.join(batch_path, model_folder)
        image_dir = os.path.join(model_dir, "images")

        if not os.path.isdir(image_dir):
            continue

        print(f"\n📁 Folder: {image_dir}")
        feature_vectors = []

        with torch.no_grad():
            for img_name in IMAGE_NAMES:
                img_path = os.path.join(image_dir, img_name)
                if not os.path.exists(img_path):
                    print(f"⚠️ Missing: {img_name}")
                    continue
                try:
                    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)  # normalize
                    feature_vectors.append(image_features.cpu().numpy())
                    print(f"✓ Encoded {img_name}")
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {e}")

        if feature_vectors:
            feature_vectors = np.vstack(feature_vectors)
            mean_embedding = np.mean(feature_vectors, axis=0)  # average of 6 embeddings

            np.save(os.path.join(model_dir, "clip_features.npy"), feature_vectors)
            np.save(os.path.join(model_dir, "clip_mean.npy"), mean_embedding)

            print(f"✅ Saved features → {model_dir}")
            print(f"   🔹 Per-image shape: {feature_vectors.shape}")
            print(f"   🔹 Mean embedding shape: {mean_embedding.shape}")
            saved_files += 1
        else:
            print("⚠️ No valid images found to encode.")

        processed_folders += 1

elapsed = time.time() - start_time
print("\n=== SUMMARY ===")
print(f"📦 Total 'images' folders processed: {processed_folders}")
print(f"💾 Total feature files saved: {saved_files}")
print(f"⏱ Total time: {elapsed:.2f} seconds")
print("✅ All done!")
