import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('C:\\Users\\Dishant bhatt\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\Dishant bhatt\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')
imga = cv2.imread('img3.jpg')
img = cv2.resize(imga,(0,0), fx=0.3, fy=0.3)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", img)
cv2.waitKey(0)