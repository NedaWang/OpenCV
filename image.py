import numpy as np
import cv2

#load in image
img = cv2.imread("opencv-logo.png")

#display image

#cv2.imshow("Image",img)
# #wait indefinitely until a user has interact with the environment.

#or initialize a named windor first to descrive how this window will actually behave.
cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
cv2.imshow("Image",img)
cv2.waitKey(0)

#write image back to file and even change in the extension variable
cv2.imwrite("output.jpg",img)
# it will return true if the code is run in terminal

# in terminal, type 'du -a' can see size of images
# the size is different since OpenCV change the encoding of the image that is saved out to disk.
