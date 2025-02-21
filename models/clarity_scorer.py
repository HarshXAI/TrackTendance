import cv2
import numpy as np
from scipy.ndimage import variance

class ClarityScorer:
    def __init__(self):
        self.blur_threshold = 100.0  # Laplacian variance threshold
        
    def compute_score(self, face_image):
        """Compute clarity score using multiple metrics"""
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
            
        # Compute blur score using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = variance(laplacian) / self.blur_threshold
        
        # Compute brightness score
        brightness_score = np.mean(gray) / 255.0
        
        # Compute contrast score
        contrast_score = np.std(gray) / 128.0
        
        # Combine scores
        clarity_score = (blur_score * 0.5 + 
                        brightness_score * 0.25 + 
                        contrast_score * 0.25)
        
        # Normalize to [0,1]
        return min(max(clarity_score, 0.0), 1.0)
