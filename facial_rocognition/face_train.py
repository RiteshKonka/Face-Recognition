import cv2 as cv
import numpy as np
from PIL import Image
import os

path = r"C:\Users\Ritesh\Desktop\Face_Recognition\dataset"
# print(path)
recognizer = cv.face.LBPHFaceRecognizer_create()
face_detector = cv.CascadeClassifier(r"C:\Users\Ritesh\Desktop\Face_Recognition\face_detection\Cascades\haarcascade_frontalface_default.xml")

def getImageAndLabels(path):
    image_paths = [os.path.join(path,name) for name in os.listdir(path)]
    faceSamples = []
    ids = []
    for image_path in image_paths:
        PIL_img = Image.open(image_path).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
    
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImageAndLabels(path)
recognizer.train(faces,np.array(ids))

recognizer.write('trainer/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))