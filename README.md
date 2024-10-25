# Enhanced Knuckle Recognition System

A sophisticated biometric authentication system that uses knuckle patterns for user identification and verification. This system employs advanced computer vision techniques and machine learning to provide accurate and reliable authentication.

## Features

- **Multiple Feature Extraction Methods**
  - ORB (Oriented FAST and Rotated BRIEF)
  - SIFT (Scale-Invariant Feature Transform)
  - Corner detection
  - Dynamic feature quality assessment

- **Advanced Image Processing**
  - Real-time hand landmark detection using MediaPipe
  - Automatic knuckle region extraction
  - Image quality enhancement and verification
  - Perspective correction and alignment

- **Robust Authentication**
  - Multi-stage verification process
  - Consecutive match confirmation
  - Adjustable similarity thresholds
  - Quality-based feature matching

- **Database Management**
  - SQLite database with proper indexing
  - Automatic database cleanup and optimization
  - Backup and restore functionality
  - Audit logging

- **User Interface**
  - Modern Tkinter-based GUI
  - Real-time video feed display
  - Individual knuckle region visualization
  - Progress indicators and status updates

## Requirements

### Software Dependencies
```
- Python 3.8+
- OpenCV (cv2)
- MediaPipe
- NumPy
- SQLite3
- Pillow (PIL)
- SciPy
- scikit-learn
```

### Hardware Requirements
- Webcam or IP camera
- Minimum 4GB RAM
- Processor with SSE4.2 support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/enhanced-knuckle-recognition.git
cd enhanced-knuckle-recognition
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python knuckle_recognition_app.py
```

2. For IP camera usage:
   - Install an IP camera app on your smartphone
   - Ensure your phone and computer are on the same network
   - Enter the camera URL in the application interface

3. Enrollment Process:
   - Click "Start Camera" to begin video feed
   - Position your hand with knuckles clearly visible
   - Click "Enroll Hand" and enter your ID
   - Maintain hand position during enrollment
   - Multiple samples will be captured automatically

4. Authentication:
   - Click "Match Hand" when your hand is in position
   - System will attempt to match against enrolled users
   - Multiple consecutive matches required for verification
   - Match results displayed with confidence scores

## Configuration

Key parameters can be adjusted in the application:

- Similarity Threshold: Controls matching sensitivity
- Motion Threshold: Affects sample capture requirements
- Match Threshold: Number of consecutive matches needed
- Feature Parameters: Quality and extraction settings

## Database Management

The system maintains a SQLite database with:

- User profiles and enrollment data
- Feature descriptors for each knuckle
- Quality scores and usage statistics
- Audit logs for system activities

Automatic maintenance includes:
- Regular cleanup of low-quality descriptors
- Deactivation of unused profiles
- Database optimization
- Automated backups

## Troubleshooting

Common issues and solutions:

1. Camera Connection:
   - Verify network connectivity
   - Check IP camera URL format
   - Ensure camera permissions are granted

2. Hand Detection:
   - Improve lighting conditions
   - Keep hand steady and parallel to camera
   - Maintain appropriate distance from camera

3. Recognition Issues:
   - Re-enroll with better lighting
   - Adjust similarity threshold
   - Clean database and remove low-quality samples

## Security Considerations

- The system uses local processing only
- No biometric data is transmitted over network
- Database is stored locally with proper access controls
- Regular automated backups
- Audit logging for all critical operations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for hand landmark detection
- OpenCV community for computer vision tools
- SQLite team for database management
- Python Tkinter developers for GUI framework

## Author

om_gholwe
numeration6@gmail.com

## Version History

- 1.0.0: Initial release
- 1.1.0: Added enhanced feature extraction
- 1.2.0: Improved database management
- 1.3.0: Added audit logging and backup features
