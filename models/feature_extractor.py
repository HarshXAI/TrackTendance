import torch
import torch.nn as nn
import cv2  # Added missing import
from facenet_pytorch import InceptionResnetV1


class FeatureExtractor:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        
    def extract_features(self, face_img):
        # Preprocess image for facenet
        if face_img.shape[0] != 160:
            face_img = cv2.resize(face_img, (160, 160))
        
        # Convert to torch tensor and normalize
        face_tensor = torch.from_numpy(face_img).float()
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
        face_tensor = (face_tensor - 127.5) / 128.0
        
        with torch.no_grad():
            embedding = self.model(face_tensor)
            
        return embedding
