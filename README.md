# Player Re-identification and Tracking System

A comprehensive solution for player tracking and re-identification in sports videos using **YOLO detection** and **custom tracking algorithms**, designed to maintain consistent player IDs even when players temporarily leave the frame.

## 🎯 Features

### Core Capabilities
- **Multi-Object Detection**: Uses YOLO model to detect players, goalkeepers, balls, and referees
- **Custom Tracking**: Advanced multi-object tracking with re-identification capabilities
- **Feature-Based Re-ID**: ResNet18-based appearance features for robust re-identification
- **Real-time Processing**: Optimized for video processing with GPU acceleration
- **Rich Visualization**: Color-coded tracking with object type classification

### Advanced Tracking Features
- **Predictive Motion**: Kalman-like motion prediction for smooth tracking
- **Adaptive Thresholds**: Dynamic similarity thresholds based on track stability
- **Feature History**: Maintains recent appearance features for stable matching
- **Inactive Track Management**: Handles temporary occlusions and re-identification

## 🚀 Quick Start

### 1. Environment Setup

**System Requirements:**
- Python 3.8+
- CUDA-compatible GPU (recommended)
- OpenCV 4.5+
- PyTorch 1.9+

**Clone and Setup:**
```bash
git clone <your-repository>
cd "Stealth mode"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `ultralytics` - YOLO model framework
- `torch` & `torchvision` - Deep learning framework
- `opencv-python` - Computer vision operations
- `numpy` & `scipy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `filterpy` - Kalman filtering
- `matplotlib` - Plotting and visualization

### 3. Prepare Your Data

**Required Files:**
- `best.pt` - Trained YOLO model (should be in root directory)
- Input video file (e.g., `15sec_input_720p.mp4`)

**Model Configuration:**
The system expects a YOLO model trained to detect:
- Class 0: Ball
- Class 1: Goalkeeper  
- Class 2: Player
- Class 3: Referee

### 4. Run the System

**Basic Usage:**
```bash
cd src
python main.py
```

**Default Configuration:**
- Input: `../15sec_input_720p.mp4`
- Output: `../output.mp4`
- Model: `../best.pt`

**Custom Configuration:**
Edit the paths in `main.py`:
```python
VIDEO_PATH = "path/to/your/input/video.mp4"
OUTPUT_PATH = "path/to/your/output/video.mp4"
MODEL_PATH = "path/to/your/model.pt"
```

### 5. Advanced Configuration

**YOLO Detection Settings:**
```python
model.overrides['conf'] = 0.2      # Global confidence threshold
model.overrides['iou'] = 0.4       # NMS IoU threshold
model.overrides['max_det'] = 50    # Maximum detections per frame
```

**Class-Specific Confidence Thresholds:**
```python
confidence_thresholds = {
    'ball': 0.2,        # Very sensitive for ball tracking
    'player': 0.25,     # Balanced for player detection
    'goalkeeper': 0.25, # Same as player
    'referee': 0.3      # Slightly higher for referees
}
```

**Tracking Parameters:**
```python
tracker = Tracker(
    max_missed=30,                    # Frames before deactivating track
    feature_similarity_thresh=0.6     # Appearance similarity threshold
)
```

## 📁 Project Structure

```
Stealth mode/
├── src/
│   ├── main.py                 # Main application entry point
│   ├── tracking.py             # Multi-object tracking with re-ID
│   ├── feature_extraction.py   # ResNet18-based feature extraction
│   └── utils.py                # Visualization utilities
├── best.pt                     # Trained YOLO model
├── 15sec_input_720p.mp4       # Example input video
├── output.mp4                  # Generated output video
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── report.md                   # Technical report
```

## 🛠 System Architecture

### 1. Detection Pipeline (`main.py`)
- **YOLO Model Loading**: Loads custom-trained model with sport-specific classes
- **Adaptive Thresholding**: Different confidence thresholds per object type
- **Batch Processing**: Processes video frames sequentially
- **Feature Integration**: Combines detection with feature extraction

### 2. Feature Extraction (`feature_extraction.py`)
- **ResNet18 Backbone**: Pre-trained ImageNet features adapted for sports
- **Multi-Transform Pipeline**: Different transforms for different object types
- **Object-Specific Sizing**: Optimized input sizes (128x64 for players, 64x64 for ball)
- **Stability Modes**: Clean vs. augmented transforms for consistent tracking

### 3. Tracking System (`tracking.py`)
- **Multi-Metric Matching**: Combines appearance, motion, and IoU similarities
- **Predictive Tracking**: Motion prediction based on velocity estimation
- **Adaptive Thresholds**: Dynamic similarity thresholds based on track age
- **Re-identification**: Handles temporary occlusions and track recovery

### 4. Visualization (`utils.py`)
- **Color-Coded Display**: Different colors for each object type
- **Real-time Statistics**: Frame count, active tracks, per-class counts
- **Track Stability**: Visual indicators for track confidence

## 🔧 Configuration Guide

### For Different Sports

**Football/Soccer:**
```python
# Higher appearance weight due to similar uniforms
tracker = Tracker(max_missed=50, feature_similarity_thresh=0.5)
confidence_thresholds['player'] = 0.2  # Lower for crowded scenes
```

### Hardware Optimization

**GPU Acceleration:**
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
extractor = FeatureExtractor(device)
```

**Memory Management:**
- Reduce `max_det` for lower memory usage
- Use smaller input resolution for faster processing
- Adjust feature history length in tracking

## 📊 Performance Monitoring

The system provides real-time feedback:
- **Frame Progress**: Shows current frame being processed
- **Active Tracks**: Number of currently tracked objects
- **Detection Counts**: Per-class detection statistics
- **Processing Speed**: Frames processed per interval

**Example Output:**
```
[INFO] Processed 30 frames, 8 active tracks
[INFO] Processed 60 frames, 12 active tracks
[INFO] Processed 90 frames, 6 active tracks
✅ Done. Output saved to: ../output.mp4
```

## 🎥 Output Features

### Video Annotations
- **Bounding Boxes**: Color-coded by object type
- **Track IDs**: Persistent IDs across frames
- **Object Classes**: Player, Goalkeeper, Ball, Referee labels
- **Statistics Overlay**: Real-time tracking statistics

### Color Scheme
- **🟡 Yellow**: Ball
- **🟢 Green**: Player
- **🟣 Pink**: Goalkeeper  
- **🟠 Orange**: Referee

## 🐛 Troubleshooting

### Common Issues

1. **No GPU Detected:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Model Loading Error:**
   - Ensure `best.pt` is in the root directory
   - Check model was trained with correct class structure

3. **Poor Tracking Performance:**
   - Adjust `feature_similarity_thresh` (lower = stricter)
   - Increase `max_missed` for longer occlusions
   - Modify confidence thresholds per object type

4. **Memory Issues:**
   - Reduce video resolution
   - Lower `max_det` parameter
   - Process fewer frames per batch

### Debug Mode
Enable detailed logging by adding:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🧪 Testing

**Test with Sample Data:**
1. Ensure `15sec_input_720p.mp4` is in root directory
2. Run `python src/main.py`
3. Check `output.mp4` for results

**Custom Testing:**
1. Replace input video path in `main.py`
2. Adjust confidence thresholds if needed
3. Monitor console output for performance metrics

## 📈 Future Enhancements

- **Real-time Streaming**: WebRTC integration for live video
- **Multiple Camera Support**: Multi-view tracking fusion
- **Advanced Re-ID**: Transformer-based appearance models
- **Sport-Specific Analytics**: Tactical analysis and statistics
- **Player Recognition**: Integration with player databases

## 📄 License

This project is designed for sports analytics and computer vision research.

---

**Built for robust sports tracking with advanced re-identification capabilities.**
