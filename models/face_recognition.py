import face_recognition
import numpy as np
import cv2
from typing import List, Tuple

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_sap_ids = []
        
    def add_student(self, sap_id: str, face_image: np.ndarray):
        """Add a student's face encoding to the system"""
        # Get face encodings from the image
        face_encodings = face_recognition.face_encodings(face_image)
        if face_encodings:
            self.known_face_encodings.append(face_encodings[0])
            self.known_sap_ids.append(sap_id)
            return True
        return False
    
    def identify_face(self, face_image: np.ndarray, tolerance=0.6) -> Tuple[str, float]:
        """Identify a face in the image and return SAP ID and confidence"""
        face_encodings = face_recognition.face_encodings(face_image)
        
        if not face_encodings:
            return None, 0.0
            
        # Compare with known faces
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encodings[0])
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]
            
            if confidence > tolerance:
                return self.known_sap_ids[best_match_index], confidence
                
        return None, 0.0
