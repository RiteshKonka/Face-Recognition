import cv2 as cv
# import os

cam = cv.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv.CascadeClassifier(r"C:\Users\Ritesh\Desktop\Face_Recognition\face_detection\Cascades\haarcascade_frontalface_default.xml")

face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count = 0
while(True):
    ret,img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count+=1
        cv.imwrite("dataset/User." + str(face_id) + "." + str(count)+".jpg",gray[y:y+h,x:x+w])
        cv.imshow('image',img)

    k = cv.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv.destroyAllWindows()

