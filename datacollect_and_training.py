import cv2
import numpy as np
import os
import boto3
from PIL import Image
from io import BytesIO

s3 = boto3.client('s3', aakid='',
                  asak='')
bucket_name = 'famouspersonsss-images'
bucket_path = 'datasets'

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face_LBPHFaceRecognizer.create()

def getImageID(s3_client, s3_bucket, s3_path):
    faces = []
    ids = []
    s3_objects = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_path)

    for obj in s3_objects.get('Contents', []):
        key = obj['Key']

        if key.endswith(".jpg"):
            response = s3_client.get_object(Bucket=s3_bucket, Key=key)
            image_bytes = response['Body'].read()
            image_np = np.array(Image.open(BytesIO(image_bytes)).convert('L'))

            Id = int(os.path.split(key)[-1].split(".")[1])

            faces.append(image_np)
            ids.append(Id)

    return ids, faces

id = input("Enter Your ID: ")
count = 0

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        image = gray[y:y+h, x:x+w]

        image_path = f'datasets/User.{id}.{count}.jpg'

        s3.upload_fileobj(BytesIO(cv2.imencode('.jpg', image)[1].tobytes()), bucket_name, image_path)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)

    if count > 10:
        break

video.release()
cv2.destroyAllWindows()
print("Data Collection and Upload to AWS S3 Done.")

IDs, facedata = getImageID(s3, bucket_name, bucket_path)
recognizer.train(facedata, np.array(IDs))
recognizer.save("Trainer.yml")

print("Training Completed.")

cv2.destroyAllWindows()
