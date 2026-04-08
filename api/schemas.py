from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class FeedbackCreate(BaseModel):
    image_id:       str
    predicted_herb: str
    correct_herb:   str
    confidence:     float
    device_id:      Optional[str] = "anonymous"
    app_version:    Optional[str] = "1.0"
    image_base64:   Optional[str] = None  
    is_new_herb:    Optional[bool] = False

class FeedbackResponse(BaseModel):
    id:             int
    image_id:       str
    predicted_herb: str
    correct_herb:   str
    confidence:     float
    created_at:     Optional[datetime] = None 

    class Config:
        from_attributes = True

class MetricCreate(BaseModel):
    herb_name:   str
    confidence:  float
    was_correct: int
    device_id:   Optional[str] = "anonymous"

class VersionResponse(BaseModel):
    model_version:    str
    min_app_version:  str
    update_available: bool
    model_url:        Optional[str] = None