import cv2
import os
import numpy as np

subjects = [""]
face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8);
    if (len(faces) == 0):
        return None, None
   
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    a=0
    
    for dir_name in dirs:
        label = a+1
        a+=1
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        subjects.append(dir_name)
        for image_name in subject_images_names:
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            #cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            #cv2.waitKey(2)
            face, rect = detect_face(image)
        
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

print("Predicting images...")

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

ret, frame = video_capture.read()
img = frame.copy()
face, rect = detect_face(img)
confirmation = []
if face is not None:
    label, confidence = face_recognizer.predict(face)
    confirmation.append(subjects[label])


while True:
    ret, frame = video_capture.read()
    face, rect = detect_face(frame)

    if face is not None:
        label, confidence = face_recognizer.predict(face)
        label_text = subjects[label]+" "+str(round(confidence,2))
        confirmation.append(subjects[label])

        if confirmation[0]==confirmation[-1]:
            if len(confirmation)==50:
                print(confirmation[0]+' is Present')
                confirmation.clear()
        else:
            confirmation.clear()
        
        if confidence<=50:
            draw_rectangle(frame, rect)
            draw_text(frame, label_text, rect[0], rect[1]-5)

    cv2.imshow('Attendance Using Facial Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print("Prediction complete")
