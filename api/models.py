from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class Feedback(Base):
    __tablename__ = "feedback"

    id              = Column(Integer, primary_key=True, index=True)
    image_id        = Column(String, index=True)
    predicted_herb  = Column(String)
    correct_herb    = Column(String)
    confidence      = Column(Float)
    device_id       = Column(String)
    app_version     = Column(String)
    image_path      = Column(String, nullable=True)  # local (dev)
    s3_key          = Column(String, nullable=True)  # S3 (prod)
    created_at      = Column(DateTime, server_default=func.now())

class Metric(Base):
    __tablename__ = "metrics"

    id           = Column(Integer, primary_key=True, index=True)
    herb_name    = Column(String, index=True)
    confidence   = Column(Float)
    was_correct  = Column(Integer)
    device_id    = Column(String)
    created_at   = Column(DateTime, server_default=func.now())