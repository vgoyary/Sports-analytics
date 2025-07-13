from scipy.spatial.distance import cosine
from collections import deque
import numpy as np

def iou(bb1, bb2):
    x1, y1, x2, y2 = bb1
    x3, y3, x4, y4 = bb2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (x4 - x3) * (y4 - y3)
    union = bb1_area + bb2_area - inter_area
    return inter_area / union if union > 0 else 0

class Track:
    def __init__(self, track_id, bbox, feature, cls=None):
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature
        self.cls = cls
        self.missed = 0
        self.trace = deque(maxlen=20)
        self.velocity = (0, 0)  # Track velocity for motion prediction
        self.feature_history = deque(maxlen=5)  # Store recent features for stability
        self.confidence_history = deque(maxlen=5)  # Track confidence over time
        self.age = 0  # Track age for stability scoring

class Tracker:
    def __init__(self, max_missed=30, feature_similarity_thresh=0.6):
        self.tracks = []
        self.next_id = 0
        self.max_missed = max_missed
        self.sim_thresh = feature_similarity_thresh
        self.inactive_tracks = []  # Store tracks that might return
        self.frame_count = 0

    def predict_bbox(self, track):
        """Predict next bounding box position based on velocity"""
        if len(track.trace) < 2:
            return track.bbox
        
        # Calculate velocity from recent positions
        current_center = self.get_center(track.bbox)
        if len(track.trace) > 1:
            prev_center = self.get_center(track.trace[-2])
            track.velocity = (current_center[0] - prev_center[0], current_center[1] - prev_center[1])
        
        # Predict next position
        x1, y1, x2, y2 = track.bbox
        w, h = x2 - x1, y2 - y1
        pred_center = (current_center[0] + track.velocity[0], current_center[1] + track.velocity[1])
        
        return (int(pred_center[0] - w//2), int(pred_center[1] - h//2), 
                int(pred_center[0] + w//2), int(pred_center[1] + h//2))
    
    def get_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate Euclidean distance between bbox centers"""
        center1 = self.get_center(bbox1)
        center2 = self.get_center(bbox2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def update(self, detections):
        self.frame_count += 1
        updated_tracks = []
        used_detections = set()

        # Sort tracks by age (older tracks get priority)
        sorted_tracks = sorted(self.tracks, key=lambda t: t.age, reverse=True)

        for track in sorted_tracks:
            if track in updated_tracks:
                continue
                
            best_match_idx = None
            best_score = float('inf')
            
            # Get predicted position
            predicted_bbox = self.predict_bbox(track)
            
            for i, (det_bbox, det_feat, det_cls) in enumerate(detections):
                if i in used_detections:
                    continue
                    
                # Only match same class
                if track.cls != det_cls:
                    continue
                
                # Calculate multiple similarity metrics
                appearance_sim = cosine(track.feature, det_feat)
                
                # Use average of recent features for more stable matching
                if len(track.feature_history) > 0:
                    avg_feature = np.mean(track.feature_history, axis=0)
                    appearance_sim = 0.7 * appearance_sim + 0.3 * cosine(avg_feature, det_feat)
                
                # Motion similarity (distance from predicted position)
                motion_dist = self.calculate_distance(predicted_bbox, det_bbox)
                motion_sim = motion_dist / 200.0  # Normalize to 0-1 range
                
                # IoU similarity
                iou_sim = 1.0 - iou(track.bbox, det_bbox)
                
                # Combined score with weights
                score = (0.5 * appearance_sim + 0.3 * motion_sim + 0.2 * iou_sim)
                
                # Age bonus for stable tracks
                age_bonus = min(track.age * 0.01, 0.1)
                score -= age_bonus
                
                # Adaptive threshold based on track stability
                adaptive_thresh = self.sim_thresh
                if track.age > 10:  # Stable track
                    adaptive_thresh *= 1.2
                elif track.age < 3:  # New track
                    adaptive_thresh *= 0.8
                
                if score < adaptive_thresh and score < best_score:
                    best_score = score
                    best_match_idx = i

            if best_match_idx is not None:
                # Update track with matched detection
                det_bbox, det_feat, det_cls = detections[best_match_idx]
                used_detections.add(best_match_idx)
                
                # Update track properties
                track.bbox = det_bbox
                track.feature = det_feat
                track.cls = det_cls
                track.missed = 0
                track.age += 1
                track.trace.append(det_bbox)
                
                # Update feature history for stability
                track.feature_history.append(det_feat)
                
                updated_tracks.append(track)

        # Check inactive tracks for re-identification
        for track in self.inactive_tracks[:]:
            if track.missed > 100:  # Skip very old tracks
                continue
                
            best_match_idx = None
            best_score = float('inf')
            
            for i, (det_bbox, det_feat, det_cls) in enumerate(detections):
                if i in used_detections or track.cls != det_cls:
                    continue
                
                # For inactive tracks, use more strict appearance matching
                if len(track.feature_history) > 0:
                    avg_feature = np.mean(track.feature_history, axis=0)
                    appearance_sim = cosine(avg_feature, det_feat)
                else:
                    appearance_sim = cosine(track.feature, det_feat)
                
                if appearance_sim < self.sim_thresh * 0.7 and appearance_sim < best_score:
                    best_score = appearance_sim
                    best_match_idx = i

            if best_match_idx is not None:
                # Reactivate track
                det_bbox, det_feat, det_cls = detections[best_match_idx]
                used_detections.add(best_match_idx)
                
                self.inactive_tracks.remove(track)
                self.tracks.append(track)
                
                track.bbox = det_bbox
                track.feature = det_feat
                track.missed = 0
                track.trace.append(det_bbox)
                track.feature_history.append(det_feat)
                
                updated_tracks.append(track)

        # Create new tracks for unmatched detections
        for i, (det_bbox, det_feat, det_cls) in enumerate(detections):
            if i not in used_detections:
                new_track = Track(self.next_id, det_bbox, det_feat, det_cls)
                self.next_id += 1
                new_track.trace.append(det_bbox)
                new_track.feature_history.append(det_feat)
                new_track.age = 1
                self.tracks.append(new_track)
                updated_tracks.append(new_track)

        # Update missed counter for tracks not matched
        for track in self.tracks:
            if track not in updated_tracks:
                track.missed += 1

        # Move tracks that missed too many frames to inactive
        active_tracks = []
        for track in self.tracks:
            if track.missed < self.max_missed:
                active_tracks.append(track)
            else:
                # Only keep stable tracks in inactive list
                if track.age > 5:
                    self.inactive_tracks.append(track)
        
        self.tracks = active_tracks
        
        # Clean up very old inactive tracks
        self.inactive_tracks = [t for t in self.inactive_tracks if t.missed < 150]
        
        return self.tracks
