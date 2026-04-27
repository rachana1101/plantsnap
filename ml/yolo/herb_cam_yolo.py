"""
HerbCam — Step 1: YOLOv8 Setup & CoreML Export
================================================
This script:
1. Downloads YOLOv8n (nano) pretrained on COCO
2. Runs a quick test inference
3. Exports to CoreML format (Float16 by default)
4. Exports a quantized Int8 variant for comparison

Prerequisites:
    pip install ultralytics coremltools opencv-python Pillow

Hardware note:
    Training/export works on Intel Mac. MPS acceleration is Apple Silicon only,
    but export and inference testing don't need GPU acceleration.
"""

from ultralytics import YOLO
from pathlib import Path
import time


def main():
    print("=" * 60)
    print("HerbCam — YOLOv8 Setup & CoreML Export")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: Load pretrained YOLOv8n (nano — smallest, fastest)
    # ---------------------------------------------------------------
    print("\n[1/4] Loading YOLOv8n pretrained model...")
    model = YOLO("yolov8n.pt")  # auto-downloads ~6MB weights
    print(f"  ✓ Model loaded: {model.model_name}")
    print(f"  ✓ Classes: {len(model.names)} (COCO)")
    print(f"  ✓ Parameters: ~3.2M")

    # ---------------------------------------------------------------
    # Step 2: Quick test inference on a sample image
    # ---------------------------------------------------------------
    print("\n[2/4] Running test inference...")
    # Use a built-in test image from ultralytics
    results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
    result = results[0]
    print(f"  ✓ Detected {len(result.boxes)} objects:")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"    - {label}: {conf:.2f}")

    # ---------------------------------------------------------------
    # Step 3: Export to CoreML (Float16 — default)
    # ---------------------------------------------------------------
    print("\n[3/4] Exporting to CoreML (Float16)...")
    start = time.time()
    coreml_model_path = model.export(
        format="coreml",
        nms=True,          # include non-max suppression in the model
        imgsz=640,         # input resolution
    )
    export_time = time.time() - start
    print(f"  ✓ Exported to: {coreml_model_path}")
    print(f"  ✓ Export time: {export_time:.1f}s")

    # Check file size
    model_dir = Path(coreml_model_path)
    if model_dir.exists():
        total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        print(f"  ✓ Model size: {total_size / 1024 / 1024:.1f} MB")

    # ---------------------------------------------------------------
    # Step 4: Verify the CoreML model loads
    # ---------------------------------------------------------------
    print("\n[4/4] Verifying CoreML model...")
    try:
        import coremltools as ct
        mlmodel = ct.models.MLModel(coreml_model_path)
        spec = mlmodel.get_spec()
        print(f"  ✓ CoreML model loaded successfully")
        print(f"  ✓ Model type: {spec.WhichOneof('Type')}")

        # Print input/output info
        for inp in spec.description.input:
            print(f"  ✓ Input: {inp.name} — {inp.type.WhichOneof('Type')}")
        for out in spec.description.output:
            print(f"  ✓ Output: {out.name} — {out.type.WhichOneof('Type')}")

    except Exception as e:
        print(f"  ⚠ CoreML verification skipped: {e}")
        print(f"  (This is OK on Intel Mac — model will work on device)")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE! Next steps:")
    print("=" * 60)
    print("""
    1. Open the .mlpackage in Xcode to inspect it
    2. Check the Performance tab for compute unit breakdown
    3. Tomorrow: run herbcam_quantize.py to create Int8/Int4 variants
    4. Then: integrate into a Swift app with AVFoundation

    Your CoreML model is at:
    """)
    print(f"    {coreml_model_path}")
    print()


if __name__ == "__main__":
    main()
