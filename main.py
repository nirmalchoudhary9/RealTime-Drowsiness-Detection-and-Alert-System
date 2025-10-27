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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import json
import time
from pathlib import Path

st.set_page_config(page_title="🚗 Drowsiness Detection", layout="wide")

# Path to dlib's predictor (place file in the same folder)
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
ALARM_WAV_PATH = "Alert.wav"  

# EAR and timing parameters
EAR_THRESHOLD = 0.25
ALARM_TRIGGER_TIME = 5.0      
BLINK_MIN_DURATION = 0.1
BLINK_MAX_DURATION = 0.4
BLINK_ALERT_THRESHOLD = 30   

# WebRTC configuration (use public STUN)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        # Load dlib model once
        if not Path(PREDICTOR_PATH).exists():
            raise FileNotFoundError(f"{PREDICTOR_PATH} not found. Please add it to the app root.")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))

        self.eyes_closed_start_time = None
        self.drowsy_frames = 0
        self.total_frames = 0
        self.blink_count = 0
        self.blink_start_time = time.time()
        self.blink_in_progress = False
        self.alarm_on = False

        self.latest_drowsiness = 0.0
        self.latest_alarm = False
        self.latest_blinks = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)

        if len(faces) > 0:
            face = faces[0]  # single face for simplicity
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            landmarks = self.predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            left_eye = landmarks[self.LEFT_EYE]
            right_eye = landmarks[self.RIGHT_EYE]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            cv2.polylines(img, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(img, [right_eye], True, (0, 255, 0), 1)

            if ear < EAR_THRESHOLD:
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = time.time()

                elapsed_time = time.time() - self.eyes_closed_start_time

                if elapsed_time > BLINK_MIN_DURATION and elapsed_time <= BLINK_MAX_DURATION and not self.blink_in_progress:
                    self.blink_count += 1
                    self.blink_in_progress = True

                if elapsed_time > BLINK_MAX_DURATION:
                    self.drowsy_frames += 1

                if elapsed_time >= ALARM_TRIGGER_TIME and not self.alarm_on:
                    self.alarm_on = True
            else:
                if self.eyes_closed_start_time is not None:
                    elapsed_time = time.time() - self.eyes_closed_start_time
                    # ignore natural short blinks
                    if not (elapsed_time > BLINK_MIN_DURATION and elapsed_time <= BLINK_MAX_DURATION):
                        self.eyes_closed_start_time = None
                self.blink_in_progress = False
                if self.alarm_on:
                    self.alarm_on = False

            self.total_frames += 1
            drowsiness_percentage = (self.drowsy_frames / self.total_frames) * 100 if self.total_frames > 0 else 0

            now = time.time()
            if now - self.blink_start_time >= 60:
                self.blink_start_time = now
                self.blink_count = 0

            if self.blink_count > BLINK_ALERT_THRESHOLD:
                cv2.putText(img, "YOU ARE FALLING ASLEEP!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(img, f"Drowsiness: {drowsiness_percentage:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(img, f"Blinks: {self.blink_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            self.latest_drowsiness = drowsiness_percentage
            self.latest_alarm = self.alarm_on
            self.latest_blinks = self.blink_count
        else:
            self.total_frames += 1
            self.latest_alarm = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🚗 Real-Time Drowsiness Detection (Streamlit + WebRTC)")
st.markdown(
    """
    This demo uses `streamlit-webrtc` to access your browser webcam and detect drowsiness (EAR-based).
    - Ensure `shape_predictor_68_face_landmarks.dat` is in the app folder.
    - Place `alert.wav` (short beep) in the app folder to hear alarms.
    """
)

cols = st.columns([3, 1])
with cols[0]:
    # Webrtc streamer
    webrtc_ctx = webrtc_streamer(
        key="drowsiness",
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=DrowsinessTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
with cols[1]:
    st.markdown("### Live Status")
    status_placeholder = st.empty()
    drowsy_placeholder = st.empty()
    blinks_placeholder = st.empty()
    audio_placeholder = st.empty()

col_start, col_stop = st.columns(2)
start_btn = col_start.button("Start Detection")
stop_btn = col_stop.button("Stop Detection")

if webrtc_ctx.video_transformer:
    transformer = webrtc_ctx.video_transformer

    if webrtc_ctx.state.playing:
        status_placeholder.success("🔴 Detection running")
    else:
        status_placeholder.info("⚪ Idle")

    drowsy_placeholder.metric("Drowsiness (%)", f"{transformer.latest_drowsiness:.2f}")
    blinks_placeholder.metric("Blink Count (minute)", f"{transformer.latest_blinks}")

    try:
        if transformer.latest_alarm and Path(ALARM_WAV_PATH).exists():
            st.warning("⚠️ Drowsiness detected! Alarm playing.")
            audio_placeholder.audio(open(ALARM_WAV_PATH, "rb").read(), format="audio/wav")
        elif transformer.latest_alarm and not Path(ALARM_WAV_PATH).exists():
            st.warning("⚠️ Drowsiness detected! (No audio file found: alert.wav)")
    except Exception:
        pass

if start_btn:
    if webrtc_ctx.state.playing:
        st.info("Already running.")
    else:
        webrtc_ctx.start()
        st.experimental_rerun()

if stop_btn:
    if webrtc_ctx.state.playing:
        webrtc_ctx.stop()
        st.success("Stopped detection.")
    else:
        st.info("Was not running.")
