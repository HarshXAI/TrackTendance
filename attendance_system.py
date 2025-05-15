import cv2
import numpy as np
import torch
from datetime import datetime
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer 
from models.clarity_scorer import ClarityScorer
from models.adaptive_lstm import AdaptiveLSTM
from utils.face_tracker import FaceTracker
from database.db_manager import DatabaseManager

class AttendanceSystem:
    def __init__(self):
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.clarity_scorer = ClarityScorer()
        self.face_tracker = FaceTracker()
        self.db = DatabaseManager()
        
        # Load student data
        self.students = self.db.get_all_students()
        self.reference_embeddings = []
        self.sap_ids = []
        
        print(f"Loaded {len(self.students)} students from database")
        
        # Extract reference embeddings
        if self.students:
            for student in self.students:
                print(f"Processing student: {student.name}, embeddings: {len(student.face_embeddings)}")
                for embedding in student.face_embeddings:
                    if isinstance(embedding, list) and len(embedding) > 0:
                        self.reference_embeddings.append(np.array(embedding))
                        self.sap_ids.append(student.sap_id)
            
            print(f"Created {len(self.reference_embeddings)} reference embeddings for matching")
        
        # Initialize LSTM model
        try:
            if len(self.students) > 0:
                self.lstm_model = AdaptiveLSTM(
                    input_size=len(student.face_embeddings[0]) if student.face_embeddings else 512,
                    hidden_size=256,
                    num_classes=len(self.students)
                )
                try:
                    # Load the trained model from best_model.pth
                    model_path = 'best_model.pth'  # Use best_model.pth in the root directory
                    
                    # Load state_dict
                    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                    
                    # Check if number of classes in the model matches current number of students
                    saved_num_classes = state_dict['classifier.weight'].size(0)
                    current_num_classes = len(self.students)
                    
                    if saved_num_classes != current_num_classes:
                        print(f"Model was trained with {saved_num_classes} students, but now there are {current_num_classes} students.")
                        print("Adapting model to current student count...")
                        
                        # Create new classifier weights with proper size
                        new_classifier_weight = torch.zeros(current_num_classes, self.lstm_model.classifier.weight.size(1))
                        new_classifier_bias = torch.zeros(current_num_classes)
                        
                        # Copy weights for existing students
                        min_classes = min(saved_num_classes, current_num_classes)
                        new_classifier_weight[:min_classes] = state_dict['classifier.weight'][:min_classes]
                        new_classifier_bias[:min_classes] = state_dict['classifier.bias'][:min_classes]
                        
                        # Update state dict with resized classifier weights
                        state_dict['classifier.weight'] = new_classifier_weight
                        state_dict['classifier.bias'] = new_classifier_bias
                    
                    # Load modified state dict
                    self.lstm_model.load_state_dict(state_dict)
                    self.lstm_model.eval()
                    
                    print(f"Successfully loaded and adapted LSTM model from {model_path}")
                    self.use_lstm = True
                except Exception as e:
                    print(f"Error loading LSTM model: {str(e)}")
                    print("Falling back to direct face matching")
                    self.use_lstm = False
            else:
                self.use_lstm = False
        except Exception as e:
            print(f"Error initializing LSTM: {str(e)}")
            self.use_lstm = False
            
        self.recognition_threshold = 0.5  # For direct matching
    
    def process_frame(self, frame):
        # Detect faces
        detections = self.face_detector.detect_faces(frame)
        if not detections:
            return [], {}
        
        # Process detections
        embeddings = []
        for det in detections:
            face_img = det['face_img']
            det['clarity_score'] = self.clarity_scorer.compute_score(face_img)
            embedding = self.face_recognizer.get_embedding(face_img)
            embeddings.append(embedding)
        
        # Update tracks
        active_tracks = self.face_tracker.update(detections, embeddings)
        
        results = {}
        for track_id, track in active_tracks.items():
            # Skip tracks that are too short
            if len(track['frames']) < 3:
                continue
                
            # Use LSTM model if available
            if self.use_lstm and len(track['frames']) >= 3:  # Need at least 10 frames for LSTM
                lstm_result = self._process_with_lstm(track)
                if lstm_result:
                    # Store the LSTM result in the results dictionary
                    results[track_id] = lstm_result
                    print(f"LSTM recognized: {lstm_result['student'].name} with confidence {lstm_result['confidence']:.4f}")
                    # Important: continue to next track after successful LSTM recognition
                    continue
            
            # Fall back to direct matching only if LSTM fails or is unavailable
            face_img = track['frames'][-1]  # Use latest frame
            current_emb = self.face_recognizer.get_embedding(face_img)
            
            # Find best match
            best_match_sap = None
            best_match_confidence = 0
            
            for i, ref_emb in enumerate(self.reference_embeddings):
                similarity = self.face_recognizer.compute_similarity(current_emb, ref_emb)
                
                if similarity > self.recognition_threshold and similarity > best_match_confidence:
                    best_match_confidence = similarity
                    best_match_sap = self.sap_ids[i]
            
            if best_match_sap:
                student = self.db.get_student_by_sap(best_match_sap)
                if student:
                    results[track_id] = {
                        'student': student,
                        'confidence': best_match_confidence,
                        'method': 'direct-matching'  # Note: Not using pre-trained FaceNet model directly
                    }
        
        return detections, results
    
    def _process_with_lstm(self, track):
        """Process face track with Adaptive LSTM"""
        try:
            # Prepare embeddings and clarity scores
            if len(track['embeddings']) < 10:
                return None
                
            # Get last 10 embeddings and clarity scores
            embeddings_list = []
            for emb in track['embeddings'][-10:]:
                if hasattr(emb, 'cpu'):
                    embeddings_list.append(emb.cpu().numpy().flatten())
                else:
                    embeddings_list.append(np.array(emb).flatten())
            
            clarity_scores = track['clarity_scores'][-10:]
            
            # Convert to tensors
            embeddings = torch.tensor(embeddings_list).float().unsqueeze(0)  # [1, 10, embed_dim]
            clarity_scores = torch.tensor(clarity_scores).float().unsqueeze(0)  # [1, 10]
            
            # Run inference
            with torch.no_grad():
                output = self.lstm_model(embeddings, clarity_scores)
                probabilities = torch.softmax(output, dim=1)
                confidence, student_idx = torch.max(probabilities, dim=1)
                confidence = confidence.item()
                student_idx = student_idx.item()
            
            # Only accept high confidence predictions
            if confidence > 0.65:
                if student_idx < len(self.students):
                    student = self.students[student_idx]
                    # Actually use LSTM for the final result when confidence is high
                    return {
                        'student': student,
                        'confidence': confidence,
                        'method': 'lstm'
                    }
        except Exception as e:
            print(f"Error in LSTM processing: {str(e)}")
        
        return None

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
                        'confidence': similarity,
                        'method': 'direct'
                    }
        
        return best_match
    
    def _combine_results(self, lstm_result, direct_result):
        """Combine LSTM and direct matching results"""
        # If either result is None, return the other
        if lstm_result is None:
            return direct_result
        if direct_result is None:
            return lstm_result
            
        # If same student identified by both methods, boost confidence
        if lstm_result['student'].sap_id == direct_result['student'].sap_id:
            # Weight LSTM higher (0.7) than direct matching (0.3)
            combined_confidence = lstm_result['confidence'] * 0.7 + direct_result['confidence'] * 0.3
            return {
                'student': lstm_result['student'],
                'confidence': combined_confidence,
                'method': 'combined'
            }
        
        # Otherwise return the one with higher weighted confidence
        lstm_weighted = lstm_result['confidence'] * 0.7
        direct_weighted = direct_result['confidence'] * 0.3
        
        return lstm_result if lstm_weighted > direct_weighted else direct_result
    
    def _compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        # Convert embeddings to the right format
        if hasattr(emb1, 'cpu'):
            emb1 = emb1.cpu().numpy().flatten()
        if isinstance(emb1, list):
            emb1 = np.array(emb1)
        
        if hasattr(emb2, 'cpu'):
            emb2 = emb2.cpu().numpy().flatten()
        if isinstance(emb2, list):
            emb2 = np.array(emb2)
            
        # Ensure embeddings are flattened
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        
        # Compute similarity
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
                    # Record attendance with the method-specific confidence
                    self.db.record_attendance(
                        str(student.id),
                        student.sap_id,
                        result['confidence'],  # This should now retain the high LSTM confidence
                        subject,
                        lecture_slot
                    )
                    attendance_log[student.sap_id] = {
                        'timestamp': datetime.now(),
                        'student': student,
                        'confidence': result['confidence'],
                        'method': result.get('method', 'unknown')
                    }
            
            # Draw results on frame
            self._draw_results(frame, detections, results)
            
            # Display frame
            # cv2.imshow('Attendance System', frame)
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
            
            # If we have results for this detection
            for track_id, result in results.items():
                # Draw student name and confidence
                if result['student'] is not None:
                    name = result['student'].name
                    method = result.get('method', 'unknown')
                    confidence = result['confidence']
                    text = f"{name} ({confidence:.2f}, {method})"
                    cv2.putText(frame, text, 
                              (bbox[0], bbox[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
