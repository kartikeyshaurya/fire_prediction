#Trying to make a simple tool that make images with videos and at specific time interval

import cv2
import os
od = 0
cam = cv2.VideoCapture("E:\\datasets\\firesense\\smoke\\neg\\testneg02.808.avi")
while(True):
    ret,frame = cam.read()

    if ret:
        name = str(od)+ ".jpg"

        if od% 29 == 0:
            print(name)
            cv2.imwrite(name, frame)
        od = od+1

    else:
        break

cam.release()
cv2.destroyAllWindows()
