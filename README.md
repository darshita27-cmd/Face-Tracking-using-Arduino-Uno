# Face-Tracking-using-Arduino-Uno
Real-time face tracking system using OpenCV and Arduino with servo motors.
This project combines computer vision and Arduino control to create a face detection and tracking system using servo motors. The system detects faces in real-time using a webcam and moves servo motors to track the detected face.

# Features
Real-time face detection using OpenCV and Haar Cascades/CVZone

Servo motor control via Arduino for pan-tilt mechanism

Visual feedback with target locking indicators

Cross-platform compatibility (Windows/Linux)

# Requirements
Hardware
Webcam

Arduino board (Uno/Mega)

Servo motors (2x)

Jumper wires

Software
Python 3.x

OpenCV (cv2)

CVZone (cvzone)

PyFirmata (for Arduino communication)

NumPy

Install the required Python packages:

bash
pip install opencv-python cvzone numpy pyfirmata
Upload StandardFirmata to your Arduino:

Open Arduino IDE

Go to File > Examples > Firmata > StandardFirmata

Upload to your Arduino board

# Usage
Connect your Arduino to the specified COM port (default is COM9)

Connect servo motors to digital pins 9 (X-axis) and 10 (Y-axis)

Run the main script:

bash
python face_tracker.py
Controls
Press 'q' to quit the application

The system will automatically detect and track faces

Configuration
You can modify these parameters in the code:

ws, hs: Video frame dimensions (default: 1280x720)

port: Arduino COM port (default: "COM9")

servo_pinX, servo_pinY: Servo control pins (default: 9 and 10)

Detection parameters in detector.findFaces()

# Troubleshooting
Camera not working:

Check if another application is using the camera

Verify camera permissions

Try changing the camera index in VideoCapture(0)

# Arduino connection issues:

Verify the correct COM port

Check StandardFirmata is uploaded

Ensure proper servo connections

# Detection problems:

Adjust lighting conditions

Modify detection parameters (scaleFactor, minNeighbors)

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.
