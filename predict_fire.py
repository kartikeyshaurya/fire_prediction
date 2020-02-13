from tensorflow.keras.models import load_model

from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os

model =  load_model("modelpath")

picture =r'C:\Users\ASUS\Desktop\github\fire_prediction\img.jpg'
output_path =r'C:\Users\ASUS\Desktop\github'
pic = cv2.imread(picture)
cv2.imshow("samplefire", pic)

CLASSES = ["Non fire", "Fire"]
img = cv2.imread('img.jpg')

#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# loop over the sampled image paths

	# load the image and clone it
image = cv2.imread(img)
output = image.copy()

	# resize the input image to be a fixed 128x128 pixels, ignoring
	# aspect ratio
image = cv2.resize(image, (128, 128))
image = image.astype("float32") / 255.0

	# make predictions on the image
preds = model.predict(np.expand_dims(image, axis=0))[0]
j = np.argmax(preds)
label = CLASSES[j]

	# draw the activity on the output frame
text = label if label == "Non-Fire" else "WARNING! Fire!"
output = imutils.resize(output, width=500)
cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)

	# write the output image to disk
filename = "12{}.png".format(i)

cv2.imwrite(output_path, output)



