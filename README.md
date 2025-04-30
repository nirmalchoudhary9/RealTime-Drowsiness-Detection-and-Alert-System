# 🚗 Real-Time Drowsiness Detection and Alert System

This project is a **real-time drowsiness detection system** designed to enhance driver safety. It uses computer vision techniques to monitor the driver's eye movements and detect signs of drowsiness. If drowsiness is detected, the system alerts the driver with a visual warning and an alarm sound.

---

## 🛠 Features

- **Real-Time Detection**: Monitors the driver's face and eyes using a webcam.
- **Drowsiness Alerts**: Triggers an alarm if the driver shows signs of drowsiness.
- **Blink Tracking**: Tracks the number of blinks and alerts if the blink count exceeds a threshold.
- **Drowsiness Percentage**: Displays the percentage of time the driver is drowsy.
- **Streamlit Interface**: Provides an interactive and user-friendly UI for starting and stopping the detection process.

---

## 🖥️ How It Works

1. **Eye Aspect Ratio (EAR)**: The system calculates the EAR to determine if the eyes are closed.
2. **Blink Detection**: Natural blinks (lasting 0.1–0.4 seconds) are ignored, while prolonged eye closure is flagged as drowsiness.
3. **Alarm Trigger**: If the eyes remain closed for more than 5 seconds, an alarm is triggered.
4. **Blink Count Alert**: If the blink count exceeds 30 blinks per minute, the system alerts the user.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A webcam for real-time video capture

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nirmalchoudhary9/RealTime-Drowsiness-Detection-and-Alert-System.git
   cd RealTime-Drowsiness-Detection-and-Alert-System
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ Usage

1. Run the Streamlit application:
   ```bash
   streamlit run main.py
   ```

2. Use the **Start Detection** button to begin monitoring.
3. Use the **Stop Detection** button to stop the process.

---

## 📂 Project Structure

```
.
├── main.py                # Main Streamlit application
├── requirements.txt       # List of dependencies
├── alert.wav              # Alarm sound file
├── shape_predictor_68_face_landmarks.dat  # Dlib's facial landmark model
└── README.md              # Project documentation
```

---

## 📋 Requirements

- OpenCV
- Dlib
- Scipy
- Pygame
- Streamlit
- Numpy

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📧 Contact

For any inquiries, please contact:
- **GitHub**: [nirmalchoudhary9](https://github.com/nirmalchoudhary9)
```
