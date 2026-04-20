from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import boto3
import logging

log = logging.getLogger("plantsnap")

DATABASE_URL = "sqlite:///./plantsnap.db"
DB_FILE = "./plantsnap.db"
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_DB_KEY = "backups/plantsnap.db"

# ── Restore DB from S3 on startup ──
def restore_db_from_s3():
    if not S3_BUCKET:
        return
    if os.path.exists(DB_FILE):
        log.info("📦 Local DB exists, skipping restore")
        return
    try:
        s3 = boto3.client("s3",
            region_name=os.getenv("AWS_REGION", "us-west-2"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        s3.download_file(S3_BUCKET, S3_DB_KEY, DB_FILE)
        log.info(f"✅ DB restored from S3: s3://{S3_BUCKET}/{S3_DB_KEY}")
    except Exception as e:
        log.info(f"📦 No S3 backup found, starting fresh: {e}")

# ── Backup DB to S3 ──
def backup_db_to_s3():
    if not S3_BUCKET:
        return
    try:
        s3 = boto3.client("s3",
            region_name=os.getenv("AWS_REGION", "us-west-2"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        s3.upload_file(DB_FILE, S3_BUCKET, S3_DB_KEY)
        log.info(f"☁️ DB backed up to S3")
    except Exception as e:
        log.error(f"❌ DB backup failed: {e}")

# Restore before creating engine!
restore_db_from_s3()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()