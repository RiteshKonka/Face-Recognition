import numpy as np
import cv2 as cv

face_casscade = cv.CascadeClassifier(r"C:\Users\Ritesh\Desktop\Face_Recognition\face_detection\Cascades\haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(1, 480)

while True:
    ret,img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_casscade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (20,20))
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]

    cv.imshow('video',img)

    k = cv.waitKey(30) &0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()