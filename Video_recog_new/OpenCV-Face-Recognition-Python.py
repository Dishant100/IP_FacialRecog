import cv2
import os

import numpy as np

subjects = ["","Dishant","elvis","Gabriel"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    if (len(faces) == 0):
        return None, None
   
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
    
    faces = []
    labels = []
    
    for dir_name in dirs:
        
        if not dir_name.startswith("s"):
            continue;
            

        label = int(dir_name.replace("s", ""))
        
        subject_dir_path = data_folder_path + "/" + dir_name
        
        subject_images_names = os.listdir(subject_dir_path)
        
        
        for image_name in subject_images_names:
            
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)
            
            #cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            #cv2.waitKey(100)
            
            face, rect = detect_face(image)
            
        
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


print("Predicting images...")

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    img = frame.copy()
    #detect face from the image
    face, rect = detect_face(img)

    if face is not None:
        #predict the image using our face recognizer 
        label, confidence = face_recognizer.predict(face)
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)

    # Display the resulting frame
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print("Prediction complete")