import os
import shutil
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────
SOURCE_DIR = "/Users/rachana_gupta/techProjects/plantsnap/data/raw"        # your 3,461 images
TRAIN_DIR  = "/Users/rachana_gupta/techProjects/plantsnap/data/train"      # 80%
VAL_DIR    = "/Users/rachana_gupta/techProjects/plantsnap/data/val"        # 20%
SPLIT      = 0.8                  # 80/20
SEED       = 42                   # reproducible split ✅

random.seed(SEED)

def split_dataset():
    source = Path(SOURCE_DIR)
    train  = Path(TRAIN_DIR)
    val    = Path(VAL_DIR)

    # Clean existing splits
    if train.exists(): shutil.rmtree(train)
    if val.exists():   shutil.rmtree(val)

    total_train = 0
    total_val   = 0

    for herb_folder in sorted(source.iterdir()):
        if not herb_folder.is_dir():
            continue

        herb_name = herb_folder.name
        images    = list(herb_folder.glob("*.jpg"))
        images   += list(herb_folder.glob("*.png"))
        images   += list(herb_folder.glob("*.jpeg"))

        if len(images) == 0:
            print(f"⚠️  No images found in {herb_name}")
            continue

        # Shuffle + split
        random.shuffle(images)
        split_idx    = int(len(images) * SPLIT)
        train_images = images[:split_idx]
        val_images   = images[split_idx:]

        # Create folders
        (train / herb_name).mkdir(parents=True, exist_ok=True)
        (val   / herb_name).mkdir(parents=True, exist_ok=True)

        # Copy images
        for img in train_images:
            shutil.copy(img, train / herb_name / img.name)
        for img in val_images:
            shutil.copy(img, val / herb_name / img.name)

        total_train += len(train_images)
        total_val   += len(val_images)

        print(f"✅ {herb_name}: "
              f"{len(train_images)} train | "
              f"{len(val_images)} val")

    print(f"\n📊 Split complete!")
    print(f"   Train: {total_train} images")
    print(f"   Val:   {total_val} images")
    print(f"   Total: {total_train + total_val} images")
    print(f"\n✅ Ready to train!")
    print(f"   python train_herbs_mlflow.py")

if __name__ == "__main__":
    split_dataset()