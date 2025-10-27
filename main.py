#Code to run locally

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

#code to run over streamlit cloud

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
import mediapipe as mp
import av
import time
import base64
import threading

st.set_page_config(page_title="Real-Time Drowsiness Detection", layout="wide")

# ----------------------------
# Constants
# ----------------------------
EAR_THRESHOLD = 0.25
ALARM_TRIGGER_TIME = 5
BLINK_MIN_DURATION = 0.1
BLINK_MAX_DURATION = 0.4
BLINK_ALERT_THRESHOLD = 30
ALERT_WAV = "alert.wav"

# ----------------------------
# Load alert sound
# ----------------------------
try:
    with open(ALERT_WAV, "rb") as f:
        ALERT_WAV_B64 = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    st.error(f"⚠️ Missing '{ALERT_WAV}'. Please add it to your project folder.")
    ALERT_WAV_B64 = None


def play_alert_html(loop=True):
    """Return HTML for looping audio playback."""
    if not ALERT_WAV_B64:
        return ""
    loop_attr = "loop" if loop else ""
    return f"""
    <audio autoplay {loop_attr}>
        <source src="data:audio/wav;base64,{ALERT_WAV_B64}" type="audio/wav">
    </audio>
    """


# ----------------------------
# Mediapipe setup
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ----------------------------
# Drowsiness Processor
# ----------------------------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.eyes_closed_start_time = None
        self.blink_in_progress = False
        self.blink_count = 0
        self.blink_start_time = time.time()
        self.drowsy_frames = 0
        self.total_frames = 0
        self.alarm_on = False
        self.alarm_play_request = False
        self.ear = 0.0
        self.active = True
        self.lock = threading.Lock()

    def reset_metrics(self):
        with self.lock:
            self.eyes_closed_start_time = None
            self.blink_in_progress = False
            self.blink_count = 0
            self.blink_start_time = time.time()
            self.drowsy_frames = 0
            self.total_frames = 0
            self.alarm_on = False
            self.alarm_play_request = False
            self.ear = 0.0

    def get_ear(self, landmarks, eye_indices, image_w, image_h):
        pts = [(landmarks[idx].x * image_w, landmarks[idx].y * image_h) for idx in eye_indices]
        pts = np.array(pts, dtype=np.float32)
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3]) + 1e-8
        return (A + B) / (2.0 * C), pts.astype(np.int32)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if not self.active:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        with self.lock:
            self.total_frames += 1

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            left_ear, left_pts = self.get_ear(face_landmarks, LEFT_EYE_IDX, w, h)
            right_ear, right_pts = self.get_ear(face_landmarks, RIGHT_EYE_IDX, w, h)
            ear = (left_ear + right_ear) / 2.0

            with self.lock:
                self.ear = ear

            cv2.polylines(img, [left_pts], True, (0, 255, 0), 1)
            cv2.polylines(img, [right_pts], True, (0, 255, 0), 1)

            # ---- Drowsiness detection ----
            if ear < EAR_THRESHOLD:
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = time.time()
                elapsed = time.time() - self.eyes_closed_start_time

                if (elapsed > BLINK_MIN_DURATION and elapsed <= BLINK_MAX_DURATION) and not self.blink_in_progress:
                    self.blink_count += 1
                    self.blink_in_progress = True

                if elapsed > BLINK_MAX_DURATION:
                    self.drowsy_frames += 1

                # Trigger alarm if eyes closed too long
                if elapsed >= ALARM_TRIGGER_TIME and not self.alarm_on:
                    self.alarm_on = True
                    self.alarm_play_request = True

                cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.blink_in_progress = False
                self.eyes_closed_start_time = None
                if self.alarm_on:
                    self.alarm_on = False
                    self.alarm_play_request = False

            # ---- High blink rate alert ----
            if self.blink_count > BLINK_ALERT_THRESHOLD:
                cv2.putText(img, "YOU ARE FALLING ASLEEP!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not self.alarm_on:
                    self.alarm_on = True
                    self.alarm_play_request = True
            else:
                if self.alarm_on and ear >= EAR_THRESHOLD:
                    self.alarm_on = False
                    self.alarm_play_request = False

        with self.lock:
            total = self.total_frames if self.total_frames > 0 else 1
            drowsiness_percentage = (self.drowsy_frames / total) * 100
            blink_count = self.blink_count
            ear_val = self.ear

        cv2.putText(img, f"Drowsiness: {drowsiness_percentage:.2f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Blinks (1m): {blink_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"EAR: {ear_val:.3f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------------
# Streamlit UI
# ----------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("🚗 Real-Time Drowsiness Detection (WebRTC)")
st.markdown("""
👁️ **Keep your eyes on the road!**  
This app monitors your eyes using your webcam and plays an alarm if you stay drowsy too long.
""")

if "show_feed" not in st.session_state:
    st.session_state["show_feed"] = False

col1, col2 = st.columns([1, 1])
with col1:
    start_btn = st.button("▶️ Start Detection")
with col2:
    stop_btn = st.button("⏹️ Stop Detection")

if start_btn:
    st.session_state["show_feed"] = True
    st.rerun()
if stop_btn:
    st.session_state["show_feed"] = False
    st.rerun()

# ---- Start webcam stream ----
if st.session_state["show_feed"]:
    webrtc_ctx = webrtc_streamer(
        key="drowsiness",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=DrowsinessProcessor,
        async_processing=True,
    )

    if webrtc_ctx and webrtc_ctx.video_processor:
        proc = webrtc_ctx.video_processor
        metrics_placeholder = st.empty()
        audio_placeholder = st.empty()

        while webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            with proc.lock:
                total = proc.total_frames if proc.total_frames > 0 else 1
                drowsiness_percentage = (proc.drowsy_frames / total) * 100
                blink_count = proc.blink_count
                ear_val = proc.ear
                alarm_req = proc.alarm_play_request
                alarm_on = proc.alarm_on

            metrics_placeholder.markdown(
                f"**EAR:** {ear_val:.3f}  \n"
                f"**Blinks (1m):** {blink_count}  \n"
                f"**Drowsiness %:** {drowsiness_percentage:.2f}%  \n"
                f"**Alarm:** {'🚨 ON' if alarm_on else 'OFF'}"
            )

            if alarm_req or alarm_on:
                audio_placeholder.markdown(play_alert_html(loop=True), unsafe_allow_html=True)
            else:
                audio_placeholder.empty()

            with proc.lock:
                proc.alarm_play_request = False

            time.sleep(0.3)
