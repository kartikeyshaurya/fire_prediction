
# tool to print out the image on a particular time interval

import cv2
import os

od = 0
random_variable = 0
one_second = 29

path = "enter the path of the video"



time_gap = 3 #enter the second that you wanna give the time gap

time_per_second = one_second * time_gap  #29 is a second in this case nearly
cam = cv2.VideoCapture(path)
while(True):
    ret,frame = cam.read()

    if ret:
        name = str(random_variable)+ ".jpg"

        if od% time_per_second == 0:
            print(name)
            cv2.imwrite(name, frame)
        od = od+1
        random_variable = random_variable+1

    else:
        break

cam.release()
cv2.destroyAllWindows()
