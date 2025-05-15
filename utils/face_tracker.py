import numpy as np
from scipy.spatial.distance import cosine

class FaceTracker:
    def __init__(self, max_disappear=30, min_similarity=0.7):
        self.next_track_id = 0
        self.tracks = {}
        self.max_disappear = max_disappear
        self.min_similarity = min_similarity
    
    def update(self, detections, embeddings):
        """Update tracks with new detections"""
        active_tracks = {}
        current_tracks = set()
        
        # Match new detections with existing tracks
        for i, (det, embedding) in enumerate(zip(detections, embeddings)):
            matched = False
            best_similarity = 0
            best_track_id = None
            
            # Compare with existing tracks
            for track_id, track in self.tracks.items():
                if track.get('disappeared', 0) > self.max_disappear:
                    continue
                
                if track['embeddings']:
                    # Compare with last embedding of track
                    track_emb = track['embeddings'][-1]
                    
                    # Safely convert embeddings to numpy arrays
                    current_emb = self._to_numpy_array(embedding)
                    previous_emb = self._to_numpy_array(track_emb)
                    
                    # Calculate similarity
                    similarity = 1 - cosine(current_emb, previous_emb)
                    
                    if similarity > self.min_similarity and similarity > best_similarity:
                        best_similarity = similarity
                        best_track_id = track_id
                        matched = True
            
            if matched:
                # Update existing track
                track = self.tracks[best_track_id]
                track['frames'].append(det['face_img'])
                track['embeddings'].append(embedding)
                track['clarity_scores'].append(det.get('clarity_score', 1.0))
                track['disappeared'] = 0
                current_tracks.add(best_track_id)
                active_tracks[best_track_id] = track
            else:
                # Create new track
                track_id = str(self.next_track_id)
                self.next_track_id += 1
                
                self.tracks[track_id] = {
                    'frames': [det['face_img']],
                    'embeddings': [embedding],
                    'clarity_scores': [det.get('clarity_score', 1.0)],
                    'disappeared': 0
                }
                current_tracks.add(track_id)
                active_tracks[track_id] = self.tracks[track_id]
        
        # Update disappeared counter for unmatched tracks
        for track_id in self.tracks:
            if track_id not in current_tracks:
                self.tracks[track_id]['disappeared'] += 1
        
        return active_tracks
    
    def _to_numpy_array(self, embedding):
        """Safely convert embedding to numpy array, regardless of type"""
        if isinstance(embedding, np.ndarray):
            return embedding.flatten()
        elif hasattr(embedding, 'cpu') and hasattr(embedding, 'numpy'):
            return embedding.cpu().numpy().flatten()
        elif hasattr(embedding, 'numpy'):
            return embedding.numpy().flatten()
        # If it's a list or other iterable, convert to numpy array
        return np.array(embedding).flatten()
