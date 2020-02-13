import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg')

#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

output = img.copy()

image = cv2.resize(img, (128, 128))
image = image.astype("float32") / 255.0

