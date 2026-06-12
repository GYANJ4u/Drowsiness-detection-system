from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyttsx3

# ---------------- VOICE (THREAD FIX) ----------------
def speak(msg):
    def run():
        engine = pyttsx3.init()
        engine.say(msg)
        engine.runAndWait()
    Thread(target=run).start()

# ---------------- FUNCTIONS ----------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    return abs(top_mean[1] - low_mean[1])

# ---------------- ARGUMENT ----------------
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0)
args = vars(ap.parse_args())

# ---------------- CONSTANTS ----------------
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 10
YAWN_THRESH = 20

COUNTER = 0
yawn_status = False

# ---------------- LOAD ----------------
print("-> Loading predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# ---------------- LOOP ----------------
while True:
    frame = vs.read()

    if frame is None:
        continue

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)

        # Draw eyes
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Draw lips
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # -------- DROWSINESS --------
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER == EYE_AR_CONSEC_FRAMES:
                speak("Wake up!")

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

        # -------- YAWN --------
        if distance > YAWN_THRESH:
            cv2.putText(frame, "YAWN ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if not yawn_status:
                yawn_status = True
                speak("Take some fresh air!")
        else:
            yawn_status = False

        # Display values
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()