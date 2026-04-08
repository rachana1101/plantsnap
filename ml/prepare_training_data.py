import sqlite3
import boto3
import os
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────
# Update these to match your environment
DB_PATH      = "plantsnap.db"                  # local or path to downloaded DB
S3_BUCKET    = "plantsnap-feedback-images"     # your S3 bucket name
ORIGINAL_DIR = "../data/raw"        # ← original downloaded images
FEEDBACK_DIR = "../data/feedback"   # ← downloaded from S3
NEW_DIR      = "../data/processed"  # ← combined for training

# ── S3 Client ─────────────────────────────────────────
# Reads AWS credentials from environment variables
# export AWS_ACCESS_KEY_ID=your_key
# export AWS_SECRET_ACCESS_KEY=your_secret
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-west-2")
)

def count_images(directory):
    """Count total images in a directory tree"""
    return sum(1 for _ in Path(directory).rglob("*.jpg"))

def count_per_class(directory):
    """Count images per herb class"""
    counts = {}
    for herb_dir in sorted(Path(directory).iterdir()):
        if herb_dir.is_dir():
            counts[herb_dir.name] = len(list(herb_dir.glob("*.jpg")))
    return counts

def prepare_dataset():
    print("\n🌿 PlantSnap Retraining Dataset Preparation")
    print("─" * 50)

    # ── Step 1: Copy original training data ──────────
    print("\n📁 Step 1: Copying original dataset...")
    if os.path.exists(NEW_DIR):
        print(f"   Removing existing: {NEW_DIR}")
        shutil.rmtree(NEW_DIR)

    shutil.copytree(ORIGINAL_DIR, NEW_DIR)
    original_count = count_images(NEW_DIR)
    print(f"   ✅ Copied {original_count} images → {NEW_DIR}")

    # ── Step 2: Pull corrections from database ────────
    print("\n📊 Step 2: Loading corrections from database...")
    conn = sqlite3.connect(DB_PATH)

    # Only use corrections where:
    # - image was saved to S3 (low confidence predictions)
    # - user actually corrected the model (different herb)
    # - not a new/unknown herb (those need expert review first)
    corrections = conn.execute("""
        SELECT correct_herb, s3_key, confidence, created_at
        FROM feedback
        WHERE s3_key IS NOT NULL
        AND correct_herb != predicted_herb
        AND is_new_herb = 0
        ORDER BY created_at DESC
    """).fetchall()

    print(f"   ✅ Found {len(corrections)} verified corrections with images")

    # ── Step 3: Download images from S3 ──────────────
    print("\n☁️  Step 3: Downloading images from S3...")
    downloaded = 0
    skipped    = 0

    for correct_herb, s3_key, confidence, created_at in corrections:
        # Sanitize herb name for folder (spaces → underscores)
        herb_folder = Path(NEW_DIR) / correct_herb.replace(" ", "_")
        herb_folder.mkdir(parents=True, exist_ok=True)

        # Unique filename to avoid overwrites
        filename   = s3_key.split("/")[-1]
        local_path = herb_folder / f"feedback_{filename}"

        try:
            s3.download_file(S3_BUCKET, s3_key, str(local_path))
            downloaded += 1
            print(f"   ✅ [{confidence:.0%} conf] {correct_herb} ← {s3_key}")
        except Exception as e:
            print(f"   ❌ Failed: {s3_key} — {e}")
            skipped += 1

    # ── Step 4: Check unknown herbs (expert review) ───
    print("\n🔍 Step 4: Checking unknown herbs (expert review queue)...")
    unknown_herbs = conn.execute("""
        SELECT correct_herb, COUNT(*) as count
        FROM feedback
        WHERE is_new_herb = 1
        GROUP BY correct_herb
        ORDER BY count DESC
    """).fetchall()

    if unknown_herbs:
        print(f"   ⚠️  {len(unknown_herbs)} unknown herbs pending expert review:")
        for herb, count in unknown_herbs:
            status = "🎯 READY" if count >= 50 else f"need {50-count} more"
            print(f"      {herb}: {count} samples — {status}")
    else:
        print("   ✅ No unknown herbs pending review")

    conn.close()

    # ── Step 5: Summary ───────────────────────────────
    total_count    = count_images(NEW_DIR)
    new_classes    = len([d for d in Path(NEW_DIR).iterdir() if d.is_dir()])
    added_images   = total_count - original_count

    print("\n" + "─" * 50)
    print("📊 Dataset Summary:")
    print(f"   Original images:      {original_count}")
    print(f"   Corrections added:    {downloaded}")
    print(f"   Failed downloads:     {skipped}")
    print(f"   Total for retraining: {total_count}")
    print(f"   Total herb classes:   {new_classes}")
    print(f"   Net new images:       {added_images}")

    if downloaded > 0:
        print(f"\n✅ Dataset ready! Run retraining with:")
        print(f"   python train_herbs_mlflow.py --data {NEW_DIR}")
    else:
        print(f"\n⚠️  No new corrections downloaded.")
        print(f"   Keep collecting feedback until you have 100+ corrections!")
        print(f"   Check: https://computer-vision-yin8.onrender.com/feedback/stats")

    print("─" * 50)

if __name__ == "__main__":
    prepare_dataset()