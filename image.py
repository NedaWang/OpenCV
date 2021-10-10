import numpy as np
import cv2

# 02_01

#load in image
img = cv2.imread("opencv-logo.png")

print(type(img))
print(img.shape)
b = img[:,:,0]
print(b)

g = ima[:,:,1]
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




# 02_02  pixel
print(type(img)) # numpy.ndarray
print(img.shape)
print(len(img)) # row pixel number
print(len(img[0])) # column pixel number
print(len(img[0][0])) # channel number 3 indicates BGR, if there is transparency layer or an alpha layer that should be 4
print(img.dtype) # unit8: unsigned integer value of 8, a maximum of 2 to 8 values in each pixel, which means the rangs is 0 to 255

print(img[10,5])
print(img[:,:,0]) # all the pixels in the first channel
print(img.size)  # total number of pixels


# 02_03 data types and structure
black = np.zeros([150,200,1],'uint8') #u -> unsigned uint8-> 0 - 255
cv2.imshow("Blace",black)
print(black[0,0,:])

ones = np.ones([150,200,3],'uint8') # one is still a small number in 0 - 255
# ones *= 150 # grey color
cv2.imshow("Ones",ones)
print(ones[0,0,:])


white = np.ones([150,200,3],'uint16') # 16 bit length image
white *= (2**16-1)
cv2.imshow("White",white)
print(white[0,0,:])


color = ones.copy() # a deep copy means completely copy all its memory space,
# meaning the two are no longer connected to each other at all.
color[:,:] = (255,0,0)
cv2.imshow("color",color)
print(color[0,0,:])

cv2.waitKey(0)
cv2.destroyAllWindows()
