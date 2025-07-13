import cv2

def draw_tracks(frame, tracks, frame_info=None):
    # Draw statistics overlay
    if frame_info:
        stats_text = f"Frame: {frame_info.get('frame_num', 0)} | Active: {len(tracks)} | " \
                    f"Players: {len([t for t in tracks if t.cls == 2])} | " \
                    f"Goalkeepers: {len([t for t in tracks if t.cls == 1])} | " \
                    f"Ball: {len([t for t in tracks if t.cls == 0])}"
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)

        # Color mapping based on class name (BGR format for OpenCV)
        color_map = {
            'ball': (0, 255, 255),      # Yellow
            'player': (0, 255, 0),      # Green  
            'goalkeeper': (255, 0, 255), # Pink
            'referee': (0, 165, 255)    # Orange
        }

        # Get class name from class index - matches actual model classes
        class_names = {
            0: 'ball',
            1: 'goalkeeper', 
            2: 'player',
            3: 'referee'
        }
        
        # Determine color based on class
        if hasattr(track, 'cls') and track.cls is not None:
            class_name = class_names.get(track.cls, 'unknown')
            color = color_map.get(class_name, (255, 0, 0))  # Default blue
        else:
            color = (255, 0, 0)  # Default blue

        # Draw bounding box with adaptive thickness based on track age
        thickness = 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Create label with ID and class
        label = f'ID {track.track_id}'
        if hasattr(track, 'cls') and track.cls is not None:
            class_name = class_names.get(track.cls, 'Unknown')
            label += f' ({class_name.capitalize()})'
        
        # Add background rectangle for better text visibility
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 8), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw text with black color for better contrast
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame
