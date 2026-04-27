"""
HerbCam — Fine-Tune YOLOv8n for Plant Detection + CoreML Export
================================================================
Run this AFTER auto_annotate_plants.py has created the dataset.

This script:
1. Fine-tunes YOLOv8n on your plant detection dataset
2. Validates on the val split
3. Exports the best model to CoreML (.mlpackage)
4. Shows comparison stats

Usage:
    python train_plant_detector.py

Training time on Intel Mac: ~1-2 hours for 30 epochs on 5K images
"""

from ultralytics import YOLO
from pathlib import Path
import time

# ── Config ─────────────────────────────────────────────
DATA_YAML = "/Users/rachana_gupta/techProjects/plantsnap/data/plant_detection/data.yaml"
EPOCHS = 30
IMG_SIZE = 640
BATCH = 8        # keep small for Intel Mac with limited RAM
MODEL = "yolov8n.pt"  # start from pretrained COCO weights


def main():
    print("=" * 60)
    print("HerbCam — Fine-Tune YOLOv8n for Plant Detection")
    print("=" * 60)

    # Verify dataset exists
    if not Path(DATA_YAML).exists():
        print(f"\n  ✗ Dataset not found at {DATA_YAML}")
        print(f"  Run auto_annotate_plants.py first!")
        return

    # Step 1: Load pretrained YOLOv8n
    print(f"\n[1/4] Loading {MODEL}...")
    model = YOLO(MODEL)
    print(f"  ✓ Loaded pretrained YOLOv8n (COCO weights)")

    # Step 2: Fine-tune
    print(f"\n[2/4] Starting fine-tuning...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Batch size: {BATCH}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  This will take ~1-2 hours on Intel Mac\n")

    start = time.time()
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device="cpu",           # Intel Mac — no MPS
        patience=10,            # early stopping if no improvement for 10 epochs
        save=True,
        project="runs/plant_detect",
        name="herbcam_v1",
        exist_ok=True,
    )
    train_time = time.time() - start
    print(f"\n  ✓ Training complete in {train_time / 60:.1f} minutes")

    # Step 3: Validate
    print(f"\n[3/4] Running validation...")
    best_model_path = Path("runs/plant_detect/herbcam_v1/weights/best.pt")
    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        metrics = best_model.val(data=DATA_YAML, imgsz=IMG_SIZE)
        print(f"  ✓ mAP50:     {metrics.box.map50:.3f}")
        print(f"  ✓ mAP50-95:  {metrics.box.map:.3f}")
        print(f"  ✓ Precision: {metrics.box.mp:.3f}")
        print(f"  ✓ Recall:    {metrics.box.mr:.3f}")
    else:
        print(f"  ⚠ Best model not found at {best_model_path}")
        best_model = model

    # Step 4: Export to CoreML
    print(f"\n[4/4] Exporting to CoreML...")
    coreml_path = best_model.export(
        format="coreml",
        nms=True,
        imgsz=IMG_SIZE,
    )
    print(f"  ✓ CoreML model: {coreml_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DONE! Plant detector ready.")
    print(f"{'=' * 60}")
    print(f"""
    Training time:  {train_time / 60:.1f} minutes
    Best model:     {best_model_path}
    CoreML model:   {coreml_path}

    Next steps:
    1. Drag the .mlpackage into your Xcode project
    2. Rename it or update LiveScanView.swift to load the new model
    3. The model now detects "plant" instead of 80 COCO classes
    4. Test on your garden plants!

    Interview talking point:
    "I auto-annotated 5K plant images using YOLO-World as a teacher,
    then fine-tuned YOLOv8n to detect plants with a single class.
    This gave me a domain-specific detector that reliably finds
    plants in any setting — not just potted plants like COCO."
    """)


if __name__ == "__main__":
    main()
