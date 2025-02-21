import cv2
import numpy as np
from datetime import datetime
from models.face_detector import FaceDetector
from models.feature_extractor import FeatureExtractor
from models.clarity_scorer import ClarityScorer
from utils.face_tracker import FaceTracker
from database.db_manager import DatabaseManager

class AttendanceSystem:
    def __init__(self):
        # Initialize pre-trained components only
        self.face_detector = FaceDetector()
        self.feature_extractor = FeatureExtractor()
        self.clarity_scorer = ClarityScorer()
        self.face_tracker = FaceTracker()
        self.db = DatabaseManager()
    
    def process_frame(self, frame):
        # Detect faces
        detections = self.face_detector.detect_faces(frame)
        if not detections:
            return [], {}
        
        # Extract features and compute clarity scores
        embeddings = []
        for det in detections:
            face_img = det['face_img']
            embedding = self.feature_extractor.extract_features(face_img)
            det['clarity_score'] = self.clarity_scorer.compute_score(face_img)
            embeddings.append(embedding)
        
        # Update tracks
        active_tracks = self.face_tracker.update(detections, embeddings)
        
        # Process each track with direct matching
        results = {}
        for track_id, track in active_tracks.items():
            if len(track['frames']) >= 5:  # Still maintain sequence length for stability
                # Use direct matching only
                best_match = self._find_best_match(track['embeddings'][-1])
                if best_match:
                    results[track_id] = best_match
        
        return detections, results
    
    def _find_best_match(self, embedding, threshold=0.6):
        """Find best matching student for given embedding"""
        students = self.db.get_all_students()
        best_similarity = 0
        best_match = None
        
        for student in students:
            for ref_embedding in student.face_embeddings:
                similarity = self._compute_similarity(
                    embedding.cpu().numpy().flatten(),
                    np.array(ref_embedding)
                )
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'student': student,
                        'confidence': similarity
                    }
        
        return best_match
    
    def _compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def process_video(self, video_source, subject=None, lecture_slot=None):
        cap = cv2.VideoCapture(video_source)
        attendance_log = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detections, results = self.process_frame(frame)
            
            # Update attendance log
            for track_id, result in results.items():
                student = result['student']
                if student.sap_id not in attendance_log:
                    # Record attendance
                    self.db.record_attendance(
                        str(student.id),
                        student.sap_id,
                        result['confidence'],
                        subject,
                        lecture_slot
                    )
                    attendance_log[student.sap_id] = {
                        'timestamp': datetime.now(),
                        'student': student,
                        'confidence': result['confidence']
                    }
            
            # Draw results on frame
            self._draw_results(frame, detections, results)
            
            # Display frame
            cv2.imshow('Attendance System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return attendance_log
    
    def _draw_results(self, frame, detections, results):
        for det in detections:
            bbox = det['bbox']
            cv2.rectangle(frame, 
                        (bbox[0], bbox[1]),
                        (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                        (0, 255, 0), 2)
