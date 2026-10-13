from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pathlib import Path
from datetime import datetime
try:
    from clip_fallback import clip_identify, is_plant
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available — /clip endpoint disabled")

import tempfile
import base64
import os
import boto3
import logging

import models
import schemas
from quality import check_quality
from database import engine, get_db

# ── Setup logging ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PlantSnap] %(levelname)s → %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("plantsnap")

# Create tables
models.Base.metadata.create_all(bind=engine)

# Local fallback folder
IMAGES_DIR = Path("feedback_images")
IMAGES_DIR.mkdir(exist_ok=True)

# S3 config
S3_BUCKET  = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
USE_S3     = bool(S3_BUCKET)

# Confidence threshold for image saving
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
log.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

s3_client = None
if USE_S3:
    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    log.info(f"S3 configured → bucket: {S3_BUCKET}")
else:
    log.info("S3 not configured → using local storage")

app = FastAPI(
    title="PlantSnap API",
    description="Feedback collection for PlantSnap herb classifier",
    version="1.0.0"
)

def save_image(image_id: str, image_base64: str):
    try:
        image_data = base64.b64decode(image_base64)
        filename   = f"{image_id}.jpg"
        size_kb    = len(image_data) / 1024

        log.info(f"💾 Saving image: {filename} ({size_kb:.1f} KB)")

        if USE_S3 and s3_client:
            s3_key = f"feedback/{filename}"
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=image_data,
                ContentType="image/jpeg"
            )
            log.info(f"✅ Saved to S3: s3://{S3_BUCKET}/{s3_key}")
            return None, s3_key
        else:
            local_path = str(IMAGES_DIR / filename)
            with open(local_path, "wb") as f:
                f.write(image_data)
            log.info(f"✅ Saved locally: {local_path}")
            return local_path, None

    except Exception as e:
        log.error(f"❌ Image save failed: {e}")
        return None, None

# ── Health check ─────────────────────────────────────
@app.get("/")
def root():
    storage = "S3" if USE_S3 else "local"
    log.info(f"Health check → storage: {storage}")
    return {
        "status":  "PlantSnap API is running! 🌿",
        "storage": storage,
        "version": "1.0.0"
    }

# ── POST /feedback ────────────────────────────────────
@app.post("/feedback", response_model=schemas.FeedbackResponse)
def submit_feedback(
    feedback: schemas.FeedbackCreate,
    db: Session = Depends(get_db)
):
    log.info("─" * 40)
    log.info(f"📥 New feedback received:")
    log.info(f"   predicted_herb: {feedback.predicted_herb}")
    log.info(f"   correct_herb:   {feedback.correct_herb}")
    log.info(f"   confidence:     {feedback.confidence:.2f}")

    # Duplicate check: the app gives every photo a unique image_id, and the offline
    # queue may resend the same feedback after a failed or timed-out request.
    existing = db.query(models.Feedback).filter(
        models.Feedback.image_id == feedback.image_id).first()
    if existing:
        log.info(f"🔁 Duplicate image_id {feedback.image_id} → returning existing record {existing.id}")
        existing.image_status = "duplicate"
        return existing

    local_path   = None
    s3_key       = None
    image_status = "not_needed"

    is_correction = feedback.correct_herb.strip().lower() != feedback.predicted_herb.strip().lower()
    if feedback.image_base64 and (is_correction or feedback.confidence < CONFIDENCE_THRESHOLD):
        # Gate 1: blurry or dark photos never reach the training data
        quality = check_quality(base64.b64decode(feedback.image_base64))
        log.info(f"🔎 Quality: sharpness {quality['sharpness']}, brightness {quality['brightness']}"
                 f" → {'pass' if quality['ok'] else 'REJECT (' + quality['reason'] + ')'}")

        # Gate 2: CLIP checks it's actually a plant
        plant_ok = quality["ok"]  # if CLIP isn't available, don't block saving
        if quality["ok"] and CLIP_AVAILABLE:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    f.write(base64.b64decode(feedback.image_base64))
                    tmp_path = f.name
                check = is_plant(tmp_path)
                plant_ok = check["is_plant"]
                log.info(f"🌱 Plant check: {check['plant_score']:.2f} "
                         f"({check['best_prompt']}) → {'pass' if plant_ok else 'REJECT'}")
            except Exception as e:
                log.error(f"⚠️ Plant check failed, saving anyway: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if not quality["ok"]:
            image_status = quality["reason"]
            log.info(f"🚫 Image not saved — {quality['reason']}")
        elif plant_ok:
            log.info("📸 Saving image — correction or low confidence, quality and plant checks passed")
            local_path, s3_key = save_image(feedback.image_id, feedback.image_base64)
            image_status = "saved" if (local_path or s3_key) else "save_failed"
        else:
            image_status = "not_plant"
            log.info("🚫 Image not saved — not a plant")

    db_feedback = models.Feedback(
        image_id       = feedback.image_id,
        predicted_herb = feedback.predicted_herb,
        correct_herb   = feedback.correct_herb,
        confidence     = feedback.confidence,
        device_id      = feedback.device_id,
        app_version    = feedback.app_version,
        image_path     = local_path,
        s3_key         = s3_key
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)  # ← this should be there

    # ← ADD THIS — force created_at to populate
    db.expire(db_feedback)
    db.refresh(db_feedback)

    log.info(f"✅ Saved → id: {db_feedback.id} (image: {image_status})")
    log.info("─" * 40)

    if image_status == "saved" and new_training_photos(db) == RETRAIN_AFTER:
        log.warning(f"🎯 {RETRAIN_AFTER} new training photos since last training — ready to retrain!")

    db_feedback.image_status = image_status
    return db_feedback

# ── POST /metrics ─────────────────────────────────────
@app.post("/metrics")
def submit_metric(
    metric: schemas.MetricCreate,
    db: Session = Depends(get_db)
):
    log.info(f"📊 Metric: {metric.herb_name} | "
             f"conf: {metric.confidence:.2f} | "
             f"correct: {metric.was_correct}")

    db_metric = models.Metric(
        herb_name   = metric.herb_name,
        confidence  = metric.confidence,
        was_correct = metric.was_correct,
        device_id   = metric.device_id
    )
    db.add(db_metric)
    db.commit()
    log.info(f"✅ Metric saved")
    return {"status": "metric recorded ✅"}


# ── GET /version ──────────────────────────────────────
# Model auto-update. ml/publish_model.py uploads a new .mlpackage to
# s3://<bucket>/models/<version>/plantsnap.mlpackage/... ; then set MODEL_VERSION
# on Render. The app compares versions on launch and downloads the files listed
# here (short-lived presigned S3 links), compiles the model on the phone and
# switches to it. The model bundled with the app is always the fallback.
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0")

def model_files(version: str) -> list:
    if not (USE_S3 and s3_client):
        return []
    prefix = f"models/{version}/plantsnap.mlpackage/"
    try:
        listing = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    except Exception as e:
        log.error(f"❌ Could not list model files: {e}")
        return []
    return [{
        "path": obj["Key"][len(prefix):],
        "size": obj["Size"],
        "url":  s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": S3_BUCKET, "Key": obj["Key"]},
                    ExpiresIn=3600),
    } for obj in listing.get("Contents", [])]

@app.get("/version", response_model=schemas.VersionResponse)
def get_version():
    files = model_files(MODEL_VERSION)
    log.info(f"📱 Version check → current: {MODEL_VERSION}, {len(files)} model file(s)")
    return {
        "model_version":    MODEL_VERSION,
        "min_app_version":  "1.0",
        "update_available": len(files) > 0,
        "model_url":        None,
        "files":            files,
    }

# ── GET /feedback/all ─────────────────────────────────
@app.get("/feedback/all")
def get_all_feedback(db: Session = Depends(get_db)):
    feedbacks = db.query(models.Feedback)\
                  .order_by(models.Feedback.created_at.desc())\
                  .all()

    log.info(f"📋 Returning {len(feedbacks)} feedback records")
    return [{
        "id":             f.id,
        "predicted_herb": f.predicted_herb,
        "correct_herb":   f.correct_herb,
        "confidence":     f.confidence,
        "is_new_herb":    f.is_new_herb if hasattr(f, 'is_new_herb') else False,
        "image_saved":    f.s3_key is not None or
                          f.image_path is not None,
        "s3_key":         f.s3_key,
        "created_at":     str(f.created_at)
    } for f in feedbacks]

# ── POST /admin/flush-db ──────────────────────────────
@app.post("/admin/flush-db")
def flush_database(db: Session = Depends(get_db)):
    """
    ⚠️ DANGER: Deletes ALL feedback and metrics!
    Use only for testing/demo reset.
    """
    feedback_count = db.query(models.Feedback).count()
    metrics_count  = db.query(models.Metric).count()

    db.query(models.Feedback).delete()
    db.query(models.Metric).delete()
    db.commit()

    log.warning(f"⚠️ Database flushed!")
    log.warning(f"   Deleted {feedback_count} feedback records")
    log.warning(f"   Deleted {metrics_count} metric records")

    return {
        "status":           "database flushed ✅",
        "deleted_feedback": feedback_count,
        "deleted_metrics":  metrics_count
    }

# ── Retrain trigger ───────────────────────────────────
# After RETRAIN_AFTER new training photos (saved since the last training run),
# the server flags that the model is ready to retrain. Training itself runs
# offline (ml/train_herbs.py); after a run, set LAST_RETRAIN_AT to its date.
RETRAIN_AFTER   = int(os.getenv("RETRAIN_AFTER", "50"))
LAST_RETRAIN_AT = os.getenv("LAST_RETRAIN_AT")  # e.g. "2026-10-13", empty = count everything

def new_training_photos(db: Session) -> int:
    q = db.query(models.Feedback).filter(
        (models.Feedback.s3_key != None) | (models.Feedback.image_path != None))
    if LAST_RETRAIN_AT:
        q = q.filter(models.Feedback.created_at >= datetime.fromisoformat(LAST_RETRAIN_AT))
    return q.count()

@app.get("/feedback/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(models.Feedback).count()
    new_photos = new_training_photos(db)
    retrain_ready = new_photos >= RETRAIN_AFTER

    return {
        "total_feedback":      total,
        "new_training_photos": new_photos,
        "retrain_after":       RETRAIN_AFTER,
        "last_retrain_at":     LAST_RETRAIN_AT,
        "retrain_ready":       retrain_ready,
        "retrain_message":     "🎯 Ready to retrain!"
                               if retrain_ready
                               else f"Need {RETRAIN_AFTER - new_photos} more training photos"
    }

@app.post("/clip")
def clip_endpoint(req: schemas.ClipRequest):
    """Second opinion when ResNet18 on the phone is unsure (app calls this when online)"""
    if not CLIP_AVAILABLE:
        return {"error": "CLIP not available on this server"}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(base64.b64decode(req.image_base64))
            tmp_path = f.name
        result = clip_identify(tmp_path)
        log.info(f"🔍 CLIP second opinion: {result['best_match']['herb']} "
                 f"({result['best_match']['confidence']:.2f})")
        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
