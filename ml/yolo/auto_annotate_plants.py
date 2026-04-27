"""
HerbCam — Auto-Annotate Plant Images with YOLO-World
=====================================================
This script:
1. Scans all images in raw/ subfolders
2. Runs YOLO-World (open-vocabulary detector) with "plant" prompt
3. Saves bounding box annotations in YOLO format
4. Splits into train/val (80/20)
5. Generates data.yaml for YOLOv8 fine-tuning

Usage:
    python auto_annotate_plants.py

Expects images at: /Users/rachana_gupta/techProjects/plantsnap/data/raw/
    raw/
    ├── basil/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── cilantro/
    │   ├── img1.jpg
    │   └── ...
    └── ...

Output:
    data/plant_detection/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────
RAW_DIR = Path("/Users/rachana_gupta/techProjects/plantsnap/data/raw")
OUTPUT_DIR = Path("/Users/rachana_gupta/techProjects/plantsnap/data/plant_detection")
CONFIDENCE_THRESHOLD = 0.15  # low threshold — we want to catch all plants
VAL_SPLIT = 0.2              # 20% for validation
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROMPT = "plant"             # single class prompt for YOLO-World


def collect_images(raw_dir):
    """Walk subfolders and collect all image paths."""
    images = []
    for subfolder in sorted(raw_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        for img_path in subfolder.iterdir():
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(img_path)
    return images


def setup_output_dirs(output_dir):
    """Create train/val directory structure."""
    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Output directory: {output_dir}")


def auto_annotate(model, images, output_dir, val_split):
    """
    Run YOLO-World on each image, save YOLO-format annotations.
    Returns stats about the annotation process.
    """
    random.seed(42)
    random.shuffle(images)

    val_count = int(len(images) * val_split)
    val_set = set(range(val_count))

    stats = {
        "total": len(images),
        "annotated": 0,
        "skipped": 0,
        "total_boxes": 0,
    }

    for idx, img_path in enumerate(images):
        split = "val" if idx in val_set else "train"

        # Run inference
        try:
            results = model.predict(
                str(img_path),
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )
        except Exception as e:
            print(f"  ⚠ Failed on {img_path.name}: {e}")
            stats["skipped"] += 1
            continue

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            stats["skipped"] += 1
            # Still copy image with empty label file (negative example)
            # This teaches the model "no plant here" for some images
            if random.random() < 0.1:  # keep 10% of negatives
                dst_img = output_dir / split / "images" / img_path.name
                dst_lbl = output_dir / split / "labels" / (img_path.stem + ".txt")
                shutil.copy2(img_path, dst_img)
                dst_lbl.touch()  # empty label file
            continue

        # Build YOLO annotation lines
        # All detections map to class 0 ("plant")
        lines = []
        for box in boxes:
            # xywhn = normalized center x, center y, width, height
            x_c, y_c, w, h = box.xywhn[0].tolist()
            lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # Copy image and save label
        # Use unique name to avoid collisions across subfolders
        parent_name = img_path.parent.name
        unique_name = f"{parent_name}_{img_path.stem}"
        dst_img = output_dir / split / "images" / f"{unique_name}{img_path.suffix}"
        dst_lbl = output_dir / split / "labels" / f"{unique_name}.txt"

        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))

        stats["annotated"] += 1
        stats["total_boxes"] += len(lines)

        # Progress
        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{len(images)}] "
                  f"annotated: {stats['annotated']}, "
                  f"boxes: {stats['total_boxes']}, "
                  f"skipped: {stats['skipped']}")

    return stats


def create_data_yaml(output_dir):
    """Generate data.yaml for YOLOv8 training."""
    data = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "plant"},
        "nc": 1,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  ✓ data.yaml saved to {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("HerbCam — Auto-Annotate Plants with YOLO-World")
    print("=" * 60)

    # Step 1: Collect images
    print(f"\n[1/4] Scanning {RAW_DIR}...")
    images = collect_images(RAW_DIR)
    print(f"  ✓ Found {len(images)} images across {len(set(p.parent.name for p in images))} classes")

    if len(images) == 0:
        print("  ✗ No images found! Check the path.")
        return

    # Step 2: Setup output
    print(f"\n[2/4] Setting up output directory...")
    if OUTPUT_DIR.exists():
        print(f"  ⚠ {OUTPUT_DIR} exists. Removing and recreating...")
        shutil.rmtree(OUTPUT_DIR)
    setup_output_dirs(OUTPUT_DIR)

    # Step 3: Load YOLO-World and set prompt
    print(f"\n[3/4] Loading YOLO-World model...")
    model = YOLO("yolov8s-worldv2.pt")  # auto-downloads ~50MB
    model.set_classes([PROMPT])          # set "plant" as the only class
    print(f"  ✓ Model loaded, prompt set to: '{PROMPT}'")
    print(f"  ✓ Confidence threshold: {CONFIDENCE_THRESHOLD}")

    # Step 4: Auto-annotate
    print(f"\n[4/4] Auto-annotating {len(images)} images...")
    print(f"  This will take a while on Intel Mac (~1-2 sec per image)")
    print(f"  Estimated time: {len(images) * 1.5 / 60:.0f}–{len(images) * 2 / 60:.0f} minutes")
    stats = auto_annotate(model, images, OUTPUT_DIR, VAL_SPLIT)

    # Step 5: Create data.yaml
    print(f"\n[5/5] Creating data.yaml...")
    yaml_path = create_data_yaml(OUTPUT_DIR)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DONE! Auto-annotation complete.")
    print(f"{'=' * 60}")
    print(f"  Total images scanned:  {stats['total']}")
    print(f"  Images with plants:    {stats['annotated']}")
    print(f"  Images skipped:        {stats['skipped']}")
    print(f"  Total bounding boxes:  {stats['total_boxes']}")
    print(f"  Avg boxes per image:   {stats['total_boxes'] / max(stats['annotated'], 1):.1f}")
    print(f"\n  Dataset ready at: {OUTPUT_DIR}")
    print(f"  data.yaml at:     {yaml_path}")
    print(f"\n  Next step: run train_plant_detector.py")


if __name__ == "__main__":
    main()
