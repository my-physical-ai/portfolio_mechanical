# Ellipse Pillar Aligner — File List

| File | Role |
|---|---|
| `app.py` | Flask server, MJPEG stream, APIs |
| `config.py` | Camera, yaw, ROI, robot calibration settings |
| `realsense_camera.py` | RealSense wrapper and depth reference capture |
| `vision_ellipse.py` | RGB ellipse detection and yaw calculation |
| `depth_roi.py` | Depth ROI height/object validation |
| `overlay.py` | Visual overlay for stream |
| `math_utils.py` | Angle normalization functions |
| `robot_interface.py` | Safe robot command preview stub |
| `test_realsense.py` | Hardware test script |
| `templates/index.html` | Web UI |
| `requirements.txt` | Python dependencies |
| `run.sh` | Server launcher |
