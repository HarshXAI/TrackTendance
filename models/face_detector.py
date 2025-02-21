import numpy as np
from mtcnn import MTCNN
import cv2
from .face_segmenter import FaceSegmenter

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
        self.segmenter = FaceSegmenter()
        
    def detect_faces(self, frame):
        faces = self.detector.detect_faces(frame)
        results = []
        
        for face in faces:
            bbox = face['box']
            confidence = face['confidence']
            keypoints = face['keypoints']
            
            # Crop face
            face_img = self.crop_face(frame, bbox)
            
            # Add segmentation
            segmented_face, mask = self.segmenter.segment_face(face_img)
            
            if segmented_face is not None:
                results.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'keypoints': keypoints,
                    'face_img': segmented_face,
                    'face_mask': mask
                })
            
        return results
    
    def crop_face(self, frame, bbox):
        x, y, w, h = bbox
        return frame[y:y+h, x:x+w]
