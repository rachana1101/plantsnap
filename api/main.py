from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pathlib import Path
from datetime import datetime
from clip_fallback import clip_identify
import tempfile
import base64
import os
import boto3
import logging

import models
import schemas
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

    local_path = None
    s3_key     = None

    if feedback.image_base64 and feedback.confidence < CONFIDENCE_THRESHOLD:
        log.info(f"📸 Saving image — confidence below threshold")
        local_path, s3_key = save_image(
            feedback.image_id,
            feedback.image_base64
        )

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

    log.info(f"✅ Saved → id: {db_feedback.id}")
    log.info("─" * 40)

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
# ── Model versions config ─────────────────────────────
# Update these after each retraining run!
MODEL_VERSION    = os.getenv("MODEL_VERSION", "1.0")
MODEL_URL        = os.getenv("MODEL_URL", None)
UPDATE_AVAILABLE = os.getenv("UPDATE_AVAILABLE", "false").lower() == "true"

@app.get("/version", response_model=schemas.VersionResponse)
def get_version():
    log.info(f"📱 Version check → current: {MODEL_VERSION}")
    return {
        "model_version":    MODEL_VERSION,
        "min_app_version":  "1.0",
        "update_available": UPDATE_AVAILABLE,
        "model_url":        MODEL_URL
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

# Add to main.py — auto-notify when threshold hit!

@app.get("/feedback/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(models.Feedback).count()
    with_images = db.query(models.Feedback).filter(
        models.Feedback.s3_key != None
    ).count()

    # Retraining recommendation
    retrain_ready = with_images >= 100

    return {
        "total_feedback":   total,
        "images_collected": with_images,
        "retrain_ready":    retrain_ready,  # ← NEW!
        "retrain_message":  "🎯 Ready to retrain!" 
                            if retrain_ready 
                            else f"Need {100 - with_images} more images"
    }

@app.post("/clip")
async def clip_endpoint(image_base64: str):
    """Fallback when ResNet confidence < 0.5"""
    # Decode base64 image to temp file
    image_bytes = base64.b64decode(image_base64)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        temp_path = f.name
    
    result = clip_identify(temp_path)
    return result