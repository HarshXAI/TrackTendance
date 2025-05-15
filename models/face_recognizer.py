import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import cv2
from typing import List, Dict, Tuple

class FaceRecognizer:
    def __init__(self):
        """Initialize with pre-trained FaceNet model"""
        # Load pre-trained FaceNet model (trained on VGGFace2 dataset)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.threshold = 0.5  # Lowered recognition threshold for better recall
        
    def recognize_face(self, face_img, reference_embeddings, student_ids):
        """Recognize a face by comparing with reference embeddings"""
        # Get face embedding
        embedding = self.get_embedding(face_img)
        
        # Compare with reference embeddings
        best_match_idx = -1
        best_similarity = 0
        
        for i, ref_emb in enumerate(reference_embeddings):
            similarity = self.compute_similarity(embedding, ref_emb)
            if similarity > self.threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i
        
        if best_match_idx != -1:
            return student_ids[best_match_idx], best_similarity
        return None, 0.0
    
    def get_embedding(self, face_img):
        """Extract face embedding from image"""
        try:
            if face_img.shape[0] != 160 or face_img.shape[1] != 160:
                face_img = cv2.resize(face_img, (160, 160))
            
            # Ensure image is in correct format (BGR)
            if face_img.max() > 1.0:
                face_img = face_img.astype(np.float32) / 255.0
            
            # Preprocess image
            face_tensor = torch.from_numpy(face_img).float()
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW
            face_tensor = (face_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
            
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error extracting embedding: {str(e)}")
            return np.zeros(512)  # Return zero embedding on error
    
    def compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        try:
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
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0.0  # Return zero similarity on error
