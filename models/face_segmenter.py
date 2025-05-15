import cv2
import numpy as np
import mediapipe as mp

class FaceSegmenter:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def segment_face(self, image):
        """Segment face from image using MediaPipe Face Mesh"""
        # Make sure image is not None and has proper dimensions
        if image is None or len(image.shape) < 2:
            return None, None
            
        height, width = image.shape[:2]
        # Convert to RGB as MediaPipe requires RGB input
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None, None
            
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Convert landmarks to points
        points = [(int(l.x * width), int(l.y * height)) for l in landmarks]
        
        # Create convex hull of face points
        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Apply mask to image
        segmented_face = cv2.bitwise_and(image, image, mask=mask)
        
        return segmented_face, mask
