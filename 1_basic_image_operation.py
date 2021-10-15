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



# 02_04 image types and color channels
color = cv2.imread("butterfly.jpg",1) # load it with full color, and 1 is th default value
cv2.imshow("Image",color)
cv2.moveWindow("Image",0,0) # window setting, showing it as top left hand corner
print(color.shape)
height, width, channels = color.shape

b,g,r = cv2.split(color) # split channels of the color image into each of its components as an individual matrix
rgb_split = np.empty([height, width*3,3],'uint8') # create an uninitialized array matrix
rgb_split[:,0:width] = cv2.merge([b,b,b])
rgb_split[:,width:width*2] = cv2.merge([g,g,g])
rgb_split[:,width*2:width*3] = cv2.merge([r,r,r])

cv2.imshow("Channels",rgb_split)
cv2.moveWindow("Channels",0,height)

# another color space: Hue Saturation and Value space
'''
Hue: the type of color in a 360 degree format
Saturation: how saturated an idividual color is
Value channel: how luminous the channel is
'''
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
hsv_split = np.concatenate((h,s,v),axis=1) # axis=1: make these three images appear side by side
cv2.imshow("Split HSV",hsv_split)
cv2.moveWindow("Split HSV",0,height*2)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 02_05 pixel manipulation and filtering
color = cv2.imread("butterfly.jpg",1)

gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
cv2.imwrite("gray.jpg",gray)

# looks verbose, but actually more efficient and faster than cv2.split operator

b = color[:,:,0]
g = color[:,:,1]
r = color[:,:,2]

rgba = cv2.merge((b,g,r,g)) # low green value would appear transparent

# JPEG images do not support image transparency
cv2.imwrite("rgba.png",rgba)

# python openCV's native image viewer does not support transparency
# so we can use OSX native image viewer



# 02_06 Blur, dilation, and erosion

'''
some filtering functions that is to pre-process or adjust an image
reduce noise or unwanted variances of an image or threshold
the goal is to make the image easy to work with
1. Gaussian Blur: smooths an image by averaging pixel values with its neighbors

2. Erosion filter: looks turn white pixels into blacl pixels, essentially eating away at the foreground.

3. Dilation filter: turn black pixels, or background pixels, into white pixels

'''

image = cv2.imread("thresh.jpg")
cv2.imshow("Original",image)

# Second parameter: how much to blur the image on each axis
# note that these values all have to be odd values
# and define how much the scale of the blur is done in each direction.

# Third parameter: Standard deviation value of kernal along horizontal direction.
blur = cv2.GaussianBlur(image,(55,5),0)
cv2.imshow("Blur",blur)


kernel = np.ones((5,5),'uint8')
dilate = cv2.dilate(image, kernel, iterations=1)
erode = cv2.erode(image, kernel, iterations=1)

cv2.imshow("Dilate",dilate)
cv2.imshow("Erode",erode)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 02_07 scale and rotate

img = cv2.imread("players.jpg",1)
cv2.imshow("Original",img)

# Scale

#(0.0): not set absolute size in pixel, and use relative factors
img_half = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
cv2.imshow("Half",img_half)
# explicitly scale the image of those dimensions
img_stretch = cv2.resize(img, (600,600))
cv2.imshow("Stretch",img_stretch)
# set interpolation mode for the scalling process
# this one is more pixelated because no imtrpolation was down between pixels during the scaling process
# it used the closest pixel compared to the source image
img_stretch_near = cv2.resize(img, (600,600), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Stretch near",img_stretch_near)


# Rotation: applying a matrix transformation

# capitalized letters are customary for matrix definitions
h, w, c = img.shape # first return value is height, second return value is width
# first variable is the origin that we want to rotate (X-axis, Y-axis)
# second variable is the rotate degree, positive value is anti-clockwise
# third variable is scale size
M = cv2.getRotationMatrix2D((w/2,h/2), -90, 1)
# output img = cv2.warpAffine(input img, affine transform, desired size of the output img)
rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("Rotated",rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 02_08 use video inputs and capture images

# request from operating system to use the camera
cap = cv2.VideoCapture(0) # 0: continuous

# a loop of video feed
while(True):   # forever loop

    ret,frame = cap.read()  # read a new frame from the video capture
    frame = cv2.resize(frame, (0,0), fx=0.5, fy = 0.5)
    cv2.imshow("Frame", frame)
    # ch is the actual key that we have captured
    ch = cv2.waitKey(1)  # run every one millisecond before the next loop
    if ch & 0xFF == ord('q'):   #  convert Q character into the key that would match the ch value
        break

cap.release()
cv2.destroyAllWindows()



# 02_09 create custom interface

# request from operating system to use the camera
cap = cv2.VideoCapture(0) # 0: continuous

# define circle attributes
color = (0,255,0)  # green
line_width = 3 # if -1: the circle would be filled instead of having a lin thickness
radius = 100
point = (0, 0)

# relocate point by clicking mouse
def click(event, x, y, flags, param):
    global point, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pressed",x,y)
        point = (x,y)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click)

# a loop of video feed
while(True):   # forever loop

    ret,frame = cap.read()  # read a new frame from the video capture
    frame = cv2.resize(frame, (0,0), fx=0.5, fy = 0.5)

    cv2.circle(frame, point, radius, color, line_width)

    cv2.imshow("Frame", frame)
    # ch is the actual key that we have captured
    ch = cv2.waitKey(1)  # run every one millisecond before the next loop
    if ch & 0xFF == ord('q'):   #  convert Q character into the key that would match the ch value
        break

cap.release()
cv2.destroyAllWindows()




# 02_10 simple drawing app

# Global variables
canvas = np.ones([500,500,3],'uint8')*255

red = (0,0,255)
blue = (255,0,0)
radius = 10
line_width = -1
point = (0, 0)
color = red
pressed = False

# click callback
def click(event, x, y, flags, param):
	global canvas, point, color, pressed
	if event == cv2.EVENT_LBUTTONDOWN:
		pressed = True
		cv2.circle(canvas, (x, y), 10, blue, -1)

	# mousemove isn't triggered after event_down
	elif event == cv2.EVENT_MOUSEMOVE and pressed == True:
		cv2.circle(canvas, (x, y), 1, color, -1)

	elif event == cv2.EVENT_LBUTTONUP:
		pressed = False
		cv2.circle(canvas, (x, y), 20, red, 1)

# window initialization and callback assignment
cv2.namedWindow("canvas")
cv2.setMouseCallback("canvas", click)

# Forever draw loop
while True:

	cv2.imshow("canvas",canvas)

	# key capture every 1ms
	ch = cv2.waitKey(1)
	if ch & 0xFF == ord('q'):
		break
	elif ch & 0xFF == ord('b'):
		color = (255,0,0)
	elif ch & 0xFF == ord('g'):
		color = (0,255,0)
	elif ch & 0xFF == ord('r'):
		color = (0,0,255)

cv2.destroyAllWindows()
