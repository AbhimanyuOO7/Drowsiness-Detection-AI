import cv2
import os
import pygame
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Pygame Mixer for Alarm
pygame.mixer.init()
sound = pygame.mixer.Sound('C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/alarm.wav')


# Load the trained model
model = load_model('C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/models/cnnCat2.h5', compile=False)

# Load Haarcascades for Face and Eye Detection
face = cv2.CascadeClassifier('C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/haar cascade files/haarcascade_righteye_2splits.xml')

# Start video capture
cap = cv2.VideoCapture(0)
score = 0

def predict_eye(eye_region):
    eye = cv2.resize(eye_region, (24, 24))
    eye = eye / 255.0
    eye = eye.reshape(1, 24, 24, 1)
    pred = np.argmax(model.predict(eye), axis=-1)
    return pred[0]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, 1.1, 5)
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    eyes_closed = 0
    for (x, y, w, h) in right_eye:
        r_eye = gray[y:y + h, x:x + w]
        if predict_eye(r_eye) == 0:
            eyes_closed += 1
        break

    for (x, y, w, h) in left_eye:
        l_eye = gray[y:y + h, x:x + w]
        if predict_eye(l_eye) == 0:
            eyes_closed += 1
        break

    if eyes_closed == 2:
        score += 1
        if score > 2:  # Alarm trigger threshold
            try:
                sound.play()
            except:
                pass
    else:
        score = max(0, score - 1)

    cv2.putText(frame, 'Score: ' + str(score), (100, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()