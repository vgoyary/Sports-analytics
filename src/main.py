import cv2
import torch
from ultralytics import YOLO
from feature_extraction import FeatureExtractor
from tracking import Tracker
from utils import draw_tracks

# ==== PATHS ====
VIDEO_PATH = "../15sec_input_720p.mp4"
OUTPUT_PATH = "../output.mp4"
MODEL_PATH = "../best.pt"

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter.fourcc(*'mp4v'), 30, (width, height))

    model = YOLO(MODEL_PATH)
    
    # Configure YOLO model for better detection
    model.overrides['conf'] = 0.2  # Lower global confidence threshold
    model.overrides['iou'] = 0.4   # Lower NMS IoU threshold for overlapping boxes
    model.overrides['agnostic_nms'] = False  # Use class-specific NMS
    model.overrides['max_det'] = 50  # Increase max detections per image
    
    print("[INFO] Loaded model class names:", model.names)
    print("[INFO] Model configured with: conf=0.2, iou=0.4, max_det=50")

    extractor = FeatureExtractor(device)
    tracker = Tracker()

    # Lower confidence thresholds for better detection
    confidence_thresholds = {
        'ball': 0.2,        # Very low threshold for ball detection
        'player': 0.25,     # Lower threshold for players
        'goalkeeper': 0.25, # Lower threshold for goalkeepers
        'referee': 0.3      # Lower threshold for referees
    }

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        results = model(frame)[0]
        detections = []

        # Process each detection
        for box in results.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class name
            class_name = model.names[cls]
            
            # Apply confidence threshold based on object type
            threshold = confidence_thresholds.get(class_name, 0.5)
            if conf < threshold:
                continue
            
            # Process all relevant object types (ball, player, goalkeeper, referee)
            if class_name in ['ball', 'player', 'goalkeeper', 'referee']:
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                
                # Extract features for tracking with object type (use stable features)
                feature = extractor.extract(cropped, object_type=class_name, stable=True)
                detections.append(((x1, y1, x2, y2), feature, cls))

        tracks = tracker.update(detections)
        
        # Create frame info for visualization
        frame_info = {
            'frame_num': frame_count,
            'detections': len(detections),
            'tracks': len(tracks)
        }
        
        frame = draw_tracks(frame, tracks, frame_info)
        out.write(frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"[INFO] Processed {frame_count} frames, {len(tracks)} active tracks")

    cap.release()
    out.release()
    print("✅ Done. Output saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
