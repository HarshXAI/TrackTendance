# Adaptive Face Recognition Attendance System 🎓

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-4.4+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An intelligent attendance system using adaptive face recognition with real-time tracking and quality assessment. Built specifically for educational institutions using SAP ID-based student identification.

## 🌟 Features

- **Real-time Face Detection & Recognition** 
  - MTCNN for robust face detection
  - Pre-trained FaceNet embeddings
  - Face quality assessment
  - Multi-face tracking

- **Student Management**
  - SAP ID-based registration
  - Multi-angle face capture
  - Automatic database updates
  - Comprehensive student profiles

- **Attendance Features**
  - Real-time attendance marking
  - Subject & lecture slot tracking
  - Confidence scoring
  - Historical attendance records

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/adaptive-face-attendance.git
   cd adaptive-face-attendance
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure MongoDB**
   ```bash
   # Start MongoDB service
   mongod
   ```

4. **Register Students**
   ```bash
   python scripts/capture_student_data.py
   ```

5. **Take Attendance**
   ```bash
   python take_attendance.py
   ```

## 📁 Project Structure

```
ipd_2/
├── models/
│   ├── face_detector.py      # MTCNN face detection
│   ├── feature_extractor.py  # FaceNet embeddings
│   └── adaptive_lstm.py      # Sequence processing
├── database/
│   ├── db_manager.py        # MongoDB interface
│   └── models.py            # Database schemas
├── utils/
│   └── face_tracker.py      # Face tracking
├── scripts/
│   └── capture_student_data.py
└── data/
    └── metadata.json        # Student information
```

## 💡 Usage Guide

### Student Registration
1. Update `data/metadata.json` with student details
2. Run capture script
3. Follow prompts to capture face data
4. Verify database entry

### Taking Attendance
1. Start the system
2. Enter subject and lecture details
3. Position camera to view students
4. Monitor real-time recognition
5. Press 'q' to end session

## 🛠 Technical Details

- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Feature Extraction**: FaceNet embeddings (Inception ResNet V1)
- **Face Tracking**: Custom implementation with embedding matching
- **Database**: MongoDB for flexible document storage
- **Quality Assessment**: Clarity scoring using multiple metrics

## 📝 Requirements

- Python 3.8+
- PyTorch 1.8+
- MongoDB 4.4+
- OpenCV
- CUDA-capable GPU (optional)

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions welcome! 

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Harsh Kanani - harshkanani80@gmail.com

