# import streamlit as st
# import cv2
# import dlib
# import numpy as np
# from scipy.spatial import distance as dist
# import pygame
# import time

# # Function to calculate Eye Aspect Ratio (EAR)
# def calculate_ear(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Constants for EAR threshold and blink duration
# EAR_THRESHOLD = 0.25
# ALARM_TRIGGER_TIME = 5  # Alarm triggers after 5 seconds of closed eyes
# BLINK_MIN_DURATION = 0.1  # Minimum duration for a blink (in seconds)
# BLINK_MAX_DURATION = 0.4  # Maximum duration for a blink (in seconds)
# BLINK_ALERT_THRESHOLD = 30  # Alert if blinks exceed this threshold in a minute

# # Initialize pygame mixer for alarm sound
# pygame.mixer.init()
# pygame.mixer.music.load("alert.wav")  # Replace with the path to your alarm sound file

# # Load Dlib's face detector and facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from dlib's model repository

# # Indices for the left and right eye landmarks
# LEFT_EYE = list(range(36, 42))
# RIGHT_EYE = list(range(42, 48))

# # Streamlit interface
# st.title("🚗 Drowsiness Detection System")
# st.markdown(
#     """
#     This application detects drowsiness in real-time using your webcam.  
#     **Features**:  
#     - Alerts when drowsiness is detected.  
#     - Tracks blink count and drowsiness percentage.  
#     """
# )

# # Start/Stop buttons
# col1, col2 = st.columns(2)
# start_detection = col1.button("Start Detection")
# stop_detection = col2.button("Stop Detection")

# if start_detection:
#     # Start video capture
#     cap = cv2.VideoCapture(0)

#     # Initialize variables
#     frame_counter = 0
#     alarm_on = False
#     eyes_closed_start_time = None
#     drowsy_frames = 0
#     total_frames = 0
#     blink_count = 0
#     blink_start_time = time.time()  # Start time for blink counting
#     blink_in_progress = False  # Flag to track if a blink is already being counted

#     # Streamlit video feed
#     stframe = st.empty()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the frame
#         faces = detector(gray)

#         if len(faces) > 0:
#             # Process only the first detected face
#             face = faces[0]

#             # Draw a rectangle around the face
#             x, y, w, h = (face.left(), face.top(), face.width(), face.height())
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#             # Get facial landmarks
#             landmarks = predictor(gray, face)
#             landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

#             # Extract eye regions
#             left_eye = landmarks[LEFT_EYE]
#             right_eye = landmarks[RIGHT_EYE]

#             # Calculate EAR for both eyes
#             left_ear = calculate_ear(left_eye)
#             right_ear = calculate_ear(right_eye)
#             ear = (left_ear + right_ear) / 2.0

#             # Visualize the eyes
#             cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
#             cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

#             # Check if EAR is below the threshold
#             if ear < EAR_THRESHOLD:
#                 if eyes_closed_start_time is None:
#                     eyes_closed_start_time = time.time()  # Start the timer for closed eyes

#                 elapsed_time = time.time() - eyes_closed_start_time

#                 # Ignore natural blinks (0.1 to 0.4 seconds)
#                 if elapsed_time > BLINK_MIN_DURATION and elapsed_time <= BLINK_MAX_DURATION and not blink_in_progress:
#                     blink_count += 1  # Count as a blink
#                     blink_in_progress = True  # Set the flag to indicate a blink is in progress

#                 # Trigger alarm if eyes are closed for more than ALARM_TRIGGER_TIME
#                 if elapsed_time > BLINK_MAX_DURATION:
#                     drowsy_frames += 1

#                 if elapsed_time >= ALARM_TRIGGER_TIME and not alarm_on:
#                     alarm_on = True
#                     pygame.mixer.music.play(-1)  # Play alarm sound in a loop

#                 # Display alert on the screen
#                 cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             else:
#                 # Reset blink flag when eyes are open
#                 blink_in_progress = False

#                 if eyes_closed_start_time is not None:
#                     elapsed_time = time.time() - eyes_closed_start_time

#                     # Ignore natural blinks (0.1 to 0.4 seconds)
#                     if elapsed_time > BLINK_MIN_DURATION and elapsed_time <= BLINK_MAX_DURATION:
#                         pass  # Do not count as drowsy frame
#                     else:
#                         eyes_closed_start_time = None  # Reset the timer when eyes are open

#                 if alarm_on:
#                     alarm_on = False
#                     pygame.mixer.music.stop()  # Stop the alarm sound

#         # Update total frame count
#         total_frames += 1

#         # Calculate drowsiness percentage
#         drowsiness_percentage = (drowsy_frames / total_frames) * 100 if total_frames > 0 else 0

#         # Check if blink count exceeds the threshold in a minute
#         current_time = time.time()
#         if current_time - blink_start_time >= 60:  # Reset blink count every minute
#             blink_start_time = current_time
#             blink_count = 0
#         if blink_count > BLINK_ALERT_THRESHOLD:
#             cv2.putText(frame, "YOU ARE FALLING ASLEEP!", (10, 120),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Display drowsiness percentage
#         cv2.putText(frame, f"Drowsiness: {drowsiness_percentage:.2f}%", (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # Display blink count
#         cv2.putText(frame, f"Blinks: {blink_count}", (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # Stream the video feed to Streamlit
#         stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

#         # Stop detection if the "Stop Detection" button is pressed
#         if stop_detection:
#             break

#     # Release resources
#     cap.release()
#     pygame.mixer.quit()
#     cv2.destroyAllWindows()

# app.py

import streamlit as st
import cv2
import numpy as np
import time
import base64
from scipy.spatial import distance as dist
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ---------------- CONFIG ----------------
EAR_THRESHOLD = 0.25
ALARM_TRIGGER_TIME = 5       # seconds eyes closed to trigger alarm
BLINK_MIN_DURATION = 0.1
BLINK_MAX_DURATION = 0.4
BLINK_ALERT_THRESHOLD = 30   # blinks per minute threshold

ALARM_FILE = "alert.wav"     # Put alert.wav in repo root

# Mediapipe face mesh indexes for eyes (6 points each)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("🚗 Real-Time Drowsiness Detection (Streamlit Cloud)")

st.markdown(
    """
    Live drowsiness detection using MediaPipe and streamlit-webrtc.
    - Uses Eye Aspect Ratio (EAR) to detect closed eyes.
    - Counts blinks and computes drowsiness %.
    - Plays an alarm in the browser when prolonged eye closure is detected.
    """
)

# Utility: load alarm audio
def get_audio_data_uri(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        return f"data:audio/wav;base64,{b64}"
    except FileNotFoundError:
        return None

ALARM_AUDIO_URI = get_audio_data_uri(ALARM_FILE)


def calculate_ear(eye_pts):
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


class DrowsinessProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.eyes_closed_start_time = None
        self.drowsy_frames = 0
        self.total_frames = 0
        self.blink_count = 0
        self.blink_start_time = time.time()
        self.blink_in_progress = False
        self.alarm_on = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        h, w = img.shape[:2]
        ear = 0.0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]

            left_eye = np.array([(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in LEFT_EYE])
            right_eye = np.array([(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in RIGHT_EYE])

            cv2.polylines(img, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(img, [right_eye], True, (0, 255, 0), 1)

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            self.total_frames += 1

            if ear < EAR_THRESHOLD:
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = time.time()

                elapsed = time.time() - self.eyes_closed_start_time
                if BLINK_MIN_DURATION < elapsed <= BLINK_MAX_DURATION and not self.blink_in_progress:
                    self.blink_count += 1
                    self.blink_in_progress = True

                if elapsed > BLINK_MAX_DURATION:
                    self.drowsy_frames += 1

                if elapsed >= ALARM_TRIGGER_TIME:
                    self.alarm_on = True
            else:
                self.eyes_closed_start_time = None
                self.blink_in_progress = False
                self.alarm_on = False

            if time.time() - self.blink_start_time >= 60:
                self.blink_start_time = time.time()
                self.blink_count = 0

            drowsiness_pct = (self.drowsy_frames / self.total_frames) * 100 if self.total_frames > 0 else 0.0
            cv2.putText(img, f"Drowsiness: {drowsiness_pct:.2f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Blinks: {self.blink_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"EAR: {ear:.3f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.alarm_on:
                cv2.putText(img, "DROWSINESS ALERT!", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        else:
            cv2.putText(img, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### Controls")
    start_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")

    st.markdown("### Alarm sound")
    if ALARM_AUDIO_URI is None:
        st.warning("alert.wav not found in repo root — alarm will not play.")
    else:
        st.success("Alarm file loaded.")

with col2:
    st.markdown("### Live Camera")
    webrtc_ctx = webrtc_streamer(
        key="drowsiness-demo",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=DrowsinessProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,  # ✅ keep this, remove in_live_mode
    )

metrics_box = st.empty()
alarm_audio_box = st.empty()


def render_metrics():
    if not webrtc_ctx or not webrtc_ctx.video_transformer:
        metrics_box.info("Waiting for live video...")
        return

    processor = webrtc_ctx.video_transformer
    if not processor:
        metrics_box.info("No video feed yet.")
        return

    drowsiness = (processor.drowsy_frames / processor.total_frames * 100
                  if processor.total_frames > 0 else 0.0)

    metrics_box.markdown(
        f"""
        **EAR:** {processor.eyes_closed_start_time if processor.eyes_closed_start_time else 0:.2f}  
        **Blinks:** {processor.blink_count}  
        **Drowsiness %:** {drowsiness:.2f}%  
        **Alarm:** {'ON' if processor.alarm_on else 'OFF'}
        """
    )

    if processor.alarm_on and ALARM_AUDIO_URI:
        alarm_audio_box.markdown(
            f'<audio autoplay><source src="{ALARM_AUDIO_URI}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )
    else:
        alarm_audio_box.empty()


refresh = st.button("Refresh metrics")
if refresh:
    render_metrics()
else:
    render_metrics()
