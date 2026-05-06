"""
HerbCam — Model Quantization Comparison
========================================
Quantizes both your CoreML models (ResNet18 + YOLOv8n) to INT8 and INT4,
and compares file sizes. Run on your Mac, then deploy variants to iPhone
to measure inference time.

Usage:
    python quantize_models.py

Expects:
    - plantsnap_v1.mlpackage (ResNet18 classifier, ~22MB)
    - best.mlpackage (YOLOv8n plant detector, ~5.9MB)
    
    Update the paths below to match your file locations.
"""

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)
from pathlib import Path
import time
import sys


# ── UPDATE THESE PATHS ─────────────────────────────────
RESNET_PATH = "/Users/rachana_gupta/techProjects/plantsnap/ml/plantsnap_v1.mlpackage"
YOLO_PATH = "/Users/rachana_gupta/techProjects/plantsnap/runs/detect/runs/plant_detect/herbcam_v1/weights/yolo.mlpackage"
OUTPUT_DIR = "/Users/rachana_gupta/techProjects/plantsnap/ml/quantized"
# ────────────────────────────────────────────────────────


def get_model_size_mb(path):
    """Get total size of an .mlpackage directory in MB."""
    path = Path(path)
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total = path.stat().st_size
    return total / 1024 / 1024


def quantize_model(model_path, output_path, dtype="int8", block_size=None):
    """Quantize a CoreML model using linear quantization."""
    print(f"  Loading {Path(model_path).name}...")
    model = ct.models.MLModel(model_path)
    
    config_params = {
        "mode": "linear_symmetric",
        "dtype": dtype,
    }
    if block_size:
        config_params["granularity"] = "per_block"
        config_params["block_size"] = block_size
    
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(**config_params)
    )
    
    print(f"  Quantizing to {dtype}" + (f" (block_size={block_size})" if block_size else "") + "...")
    start = time.time()
    compressed = linear_quantize_weights(model, config=config)
    elapsed = time.time() - start
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to {output_path}...")
    compressed.save(output_path)
    
    return elapsed


def quantize_one_model(name, model_path, output_dir):
    """Run all quantization variants for one model."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    if not Path(model_path).exists():
        print(f"  ✗ Model not found at {model_path}")
        print(f"  Update the path at the top of this script!")
        return None
    
    # Baseline
    baseline_size = get_model_size_mb(model_path)
    print(f"\n  [Baseline] Float16: {baseline_size:.1f} MB")
    
    results = {
        "name": name,
        "float16_mb": baseline_size,
    }
    
    # INT8
    print(f"\n  [INT8 Quantization]")
    stem = Path(model_path).stem
    int8_path = f"{output_dir}/{stem}_int8.mlpackage"
    try:
        elapsed = quantize_model(model_path, int8_path, dtype="int8")
        int8_size = get_model_size_mb(int8_path)
        reduction = (1 - int8_size / baseline_size) * 100
        print(f"  ✓ Size: {int8_size:.1f} MB ({reduction:.0f}% reduction)")
        print(f"  ✓ Time: {elapsed:.1f}s")
        results["int8_mb"] = int8_size
        results["int8_reduction"] = reduction
    except Exception as e:
        print(f"  ⚠ INT8 failed: {e}")
        results["int8_mb"] = None
    
    # INT4 (block-wise)
    print(f"\n  [INT4 Block-wise Quantization]")
    int4_path = f"{output_dir}/{stem}_int4.mlpackage"
    try:
        elapsed = quantize_model(model_path, int4_path, dtype="int4",
                                 block_size=32)
        int4_size = get_model_size_mb(int4_path)
        reduction = (1 - int4_size / baseline_size) * 100
        print(f"  ✓ Size: {int4_size:.1f} MB ({reduction:.0f}% reduction)")
        print(f"  ✓ Time: {elapsed:.1f}s")
        results["int4_mb"] = int4_size
        results["int4_reduction"] = reduction
    except Exception as e:
        print(f"  ⚠ INT4 failed: {e}")
        results["int4_mb"] = None
    
    return results


def print_comparison(all_results):
    """Print the final comparison table."""
    print(f"\n\n{'='*70}")
    print(f"  COMPARISON TABLE — save this for the interview!")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Float16':>10} {'INT8':>10} {'INT4':>10}")
    print(f"  {'-'*55}")
    
    for r in all_results:
        if r is None:
            continue
        f16 = f"{r['float16_mb']:.1f} MB"
        int8 = f"{r.get('int8_mb', 0):.1f} MB" if r.get('int8_mb') else "failed"
        int4 = f"{r.get('int4_mb', 0):.1f} MB" if r.get('int4_mb') else "failed"
        print(f"  {r['name']:<25} {f16:>10} {int8:>10} {int4:>10}")
    
    print(f"\n  {'Model':<25} {'INT8 ↓':>10} {'INT4 ↓':>10}")
    print(f"  {'-'*45}")
    for r in all_results:
        if r is None:
            continue
        int8_r = f"{r.get('int8_reduction', 0):.0f}%" if r.get('int8_reduction') else "—"
        int4_r = f"{r.get('int4_reduction', 0):.0f}%" if r.get('int4_reduction') else "—"
        print(f"  {r['name']:<25} {int8_r:>10} {int4_r:>10}")
    
    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  NEXT STEP: Deploy each variant to your iPhone and measure   │
  │  inference time across all compute unit modes:               │
  │                                                              │
  │  Auto (ANE+GPU+CPU) | CPU+GPU | CPU+NE | CPU only           │
  │                                                              │
  │  Add those numbers to this table for the interview.          │
  │  Use the compute unit picker in your HerbCam app, or         │
  │  generate Xcode Performance Reports for each variant.        │
  └──────────────────────────────────────────────────────────────┘
    """)


def main():
    print("=" * 60)
    print("  HerbCam — Model Quantization Comparison")
    print("=" * 60)
    
    all_results = []
    
    # ResNet18 classifier
    r1 = quantize_one_model("ResNet18 (classifier)", RESNET_PATH, OUTPUT_DIR)
    all_results.append(r1)
    
    # YOLOv8n detector
    r2 = quantize_one_model("YOLOv8n (detector)", YOLO_PATH, OUTPUT_DIR)
    all_results.append(r2)
    
    # Print comparison
    print_comparison(all_results)


if __name__ == "__main__":
    main()