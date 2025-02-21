from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from bson import ObjectId

class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            if isinstance(v, ObjectId):
                return str(v)
            elif isinstance(v, str):
                ObjectId(v)
                return v
        except Exception:
            raise ValueError('Invalid ObjectId')
        return v

class StudentModel(BaseModel):
    id: Optional[str] = Field(alias='_id', default=None)  # Changed to str
    sap_id: str
    name: str
    year: int
    branch: str
    gmail: str
    face_embeddings: List[List[float]]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

    @validator('id', pre=True)
    def convert_objectid(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        return v

class AttendanceRecord(BaseModel):
    id: Optional[str] = Field(alias='_id', default=None)  # Make id optional with default None
    student_id: str
    sap_id: str
    timestamp: datetime
    confidence: float
    subject: Optional[str] = None
    lecture_slot: Optional[str] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

    @validator('id', pre=True)
    def convert_objectid(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        return v
