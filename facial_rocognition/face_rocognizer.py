import cv2 as cv
import numpy as np

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read(r"C:\Users\Ritesh\Desktop\Face_Recognition\trainer\trainer.yml")
casscadePath = r"C:\Users\Ritesh\Desktop\Face_Recognition\face_detection\Cascades\haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(casscadePath)

font = cv.FONT_HERSHEY_SIMPLEX
id = 0

names = ['Hastansh','Rachit','Aditya','Ritesh','Keya','Varun']

cam = cv.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True :
    ret,img = cam.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(minW), int(minH)))

    for (x,y,w,h) in faces :
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence<100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else :
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        cv.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
        cv.putText(img,str(confidence),(x-10,y-10),font,1,(255,255,255),1)
    cv.imshow('camera',img)
    k = cv.waitKey(0) & 0xff # Press 'ESC' for exiting video
    if k == 27: 
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv.destroyAllWindows()

