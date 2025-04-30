import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame
import time

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for EAR threshold and blink duration
EAR_THRESHOLD = 0.25
ALARM_TRIGGER_TIME = 5  # Alarm triggers after 5 seconds of closed eyes
BLINK_MIN_DURATION = 0.1  # Minimum duration for a blink (in seconds)
BLINK_MAX_DURATION = 0.4  # Maximum duration for a blink (in seconds)
BLINK_ALERT_THRESHOLD = 30  # Alert if blinks exceed this threshold in a minute

# Initialize frame counter, alarm status, and drowsiness tracking
frame_counter = 0
alarm_on = False
eyes_closed_start_time = None
drowsy_frames = 0
total_frames = 0
blink_count = 0
blink_start_time = time.time()  # Start time for blink counting
blink_in_progress = False  # Flag to track if a blink is already being counted

# Initialize pygame mixer for alarm sound
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")  # Replace with the path to your alarm sound file

# Load Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from dlib's model repository

# Indices for the left and right eye landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    if len(faces) > 0:
        # Process only the first detected face
        face = faces[0]

        # Draw a rectangle around the face
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get facial landmarks
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract eye regions
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Visualize the eyes
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        # Check if EAR is below the threshold
        if ear < EAR_THRESHOLD:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()  # Start the timer for closed eyes

            elapsed_time = time.time() - eyes_closed_start_time

            # Ignore natural blinks (0.1 to 0.4 seconds)
            if elapsed_time > BLINK_MIN_DURATION and elapsed_time <= BLINK_MAX_DURATION and not blink_in_progress:
                blink_count += 1  # Count as a blink
                blink_in_progress = True  # Set the flag to indicate a blink is in progress

            # Trigger alarm if eyes are closed for more than ALARM_TRIGGER_TIME
            if elapsed_time > BLINK_MAX_DURATION:
                drowsy_frames += 1

            if elapsed_time >= ALARM_TRIGGER_TIME and not alarm_on:
                alarm_on = True
                pygame.mixer.music.play(-1)  # Play alarm sound in a loop

            # Display alert on the screen
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Reset blink flag when eyes are open
            blink_in_progress = False

            if eyes_closed_start_time is not None:
                elapsed_time = time.time() - eyes_closed_start_time

                # Ignore natural blinks (0.1 to 0.4 seconds)
                if elapsed_time > BLINK_MIN_DURATION and elapsed_time <= BLINK_MAX_DURATION:
                    pass  # Do not count as drowsy frame
                else:
                    eyes_closed_start_time = None  # Reset the timer when eyes are open

            if alarm_on:
                alarm_on = False
                pygame.mixer.music.stop()  # Stop the alarm sound

    # Update total frame count
    total_frames += 1

    # Calculate drowsiness percentage
    drowsiness_percentage = (drowsy_frames / total_frames) * 100 if total_frames > 0 else 0

    # Check if blink count exceeds the threshold in a minute
    current_time = time.time()
    if current_time - blink_start_time >= 60:  # Reset blink count every minute
        blink_start_time = current_time
        blink_count = 0
    if blink_count > BLINK_ALERT_THRESHOLD:
        cv2.putText(frame, "YOU ARE FALLING ASLEEP!", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display drowsiness percentage
    cv2.putText(frame, f"Drowsiness: {drowsiness_percentage:.2f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display blink count
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()