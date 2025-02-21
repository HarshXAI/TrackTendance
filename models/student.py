from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Student:
    sap_id: str
    name: str
    year: int
    branch: str
    gmail: str
    face_embedding: np.ndarray = None
    
    def to_dict(self):
        return {
            'sap_id': self.sap_id,
            'name': self.name,
            'year': self.year,
            'branch': self.branch,
            'gmail': self.gmail,
            'face_embedding': self.face_embedding
        }
    
    @staticmethod
    def from_dict(data):
        return Student(**data)
