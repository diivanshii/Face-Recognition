import cv2
import boto3
import numpy as np
from PIL import Image
from io import BytesIO

s3 = boto3.client('s3', aakid='',
                  asak='')

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read("Trainer.yml")

name_list = ["", "Matched", "Sanchit", "Pritam Sir"]

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        serial, conf = recognizer.predict(face_roi)

        if conf < 50:
            cv2.putText(frame, name_list[serial], (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        else:
            cv2.putText(frame, "Access Denied", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("Facial Recognition Monitor", frame)
    k = cv2.waitKey(1)

    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
print("Face Recognition Done..................")
