# 03_02 simple thresholding
import numpy as np
import cv2

'''
Segmentation can be done through different ways,
but the typical output is a binary image.
A binary image is sth that has values of zero or one,
ones are pieces of image we want to use, 0 is everything else.
Binary images are pure, non alias black and white images.
They act as a mask for the areas of thr sourced image.
'''

# perform our own simple thresholding
bw = cv2.imread('detect_blob.png', 0)  # 0: load as black and white image
height, width = bw.shape[0:2]   # get first two values
cv2.imshow("Black and White", bw)
# initialize binary variable
binary = np.zeros([height,width,1], 'uint8')
# set a threshold
thresh = 85
for row in range(0, height):
    for col in range(0, width):
        if bw[row][col] > thresh:
            binary[row][col] = 255

cv2.imshow("Slow Binary", binary)  # this method is slower


# openCV build-in segmentation method
ret, new_thresh = cv2.threshold(bw, thresh, 255, cv2.THRESH_BINARY)
cv2.imshow("CV Threshold", new_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 03_03 adaptive thresholding
'''
Simple thresholding has limitation, such as when there's uneven lighting in an image.
This is where adaptive thresholding comes to the rescue, which increase the versatility of image thresholding operations.
Instead of taking a simple global value as a threshold comparison, adaptive thresholding will look in its local neighborhood of the image to determine whether a relative threshold is met.
'''

img = cv2.imread("sudoku.png", 0)
cv2.imshow("Original", img)

ret, thresh_basic = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
cv2.imshow("Basic Binary", thresh_basic)
# 255: maximum pixel value
# 115: how far or what th localization of where the adaptive threshoding will act over
# 1: a mean subtraction from the end result
thres_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow("Adaptive threshold", thres_adapt)

kernel = np.ones((3,3),'uint8')
dilate = cv2.dilate(thres_adapt, kernel, iterations=1)
cv2.imshow("dilate", dilate)


cv2.waitKey(0)
cv2.destroyAllWindows()




# 03_04 skin detection
img = cv2.imread('faces.jpeg',1)
img = cv2.resize(img, (500,500))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = hsv[:,:,0]  # hue: circular parameter, red is indicated by low numbers such as zero, or very high number such as 255. Blue and green are in the middle, represented as the more gray values.
s = hsv[:,:,1]  # saturation
v = hsv[:,:,2]  # intensity value
# display side by side
hsv_split = np.concatenate((h,s,v), axis=1)
cv2.imshow("hsv", hsv_split)

ret, min_sat = cv2.threshold(s,40,255, cv2.THRESH_BINARY)
cv2.imshow("Sat Filter", min_sat)

ret, max_hue = cv2.threshold(h,15,255, cv2.THRESH_BINARY_INV)  # inverse of the normal order of the threshold, so make 0-15 to white
cv2.imshow("Hue Filter", max_hue)

final = cv2.bitwise_and(min_sat, max_hue)
cv2.imshow("Final", final)
cv2.imshow("Original", img)

cv2.waitKey(0)
cv2.destroyAllWindows()



# 03_06 contour
# print(cv2.__version__)  # 4.0.1
# contour is an iterative energy reduction algorithm,
img = cv2.imread('detect_blob.png',1)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,155,1)
cv2.imshow("Binary", thresh)

#contours: actually list of individual contours and recall that each contour is a list of points which describe a parameter
#hierarchy: a parant-child relationship of all the contours. A child would be if one contour is enclosed by another.
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = img.copy()
index = -1   # -1: draw all contours
thickness = 4
color = (255, 0 , 255)

cv2.drawContours(img2, contours, index, color, thickness)
cv2.imshow("Contours", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 03_07 area, perimeter, centroid
img = cv2.imread('detect_blob.png',1)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow("Binary", thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = img.copy()
index = -1
thickness = 4
color = (255, 0, 255)

objects = np.zeros([img.shape[0], img.shape[1],3], 'uint8')
for c in contours:
	# [c]: a list of a single contour that the one we're using right now
	# -1: draw all the contours, which is just going to be the on countour on the list
	# -1: fill in the color
	cv2.drawContours(objects, [c], -1, color, -1)

	area = cv2.contourArea(c)
	perimeter = cv2.arcLength(c, True) # true: closed contour

	M = cv2.moments(c) # image moment
	# calculate centroid centre
	cx = int( M['m10']/M['m00'])
	cy = int( M['m01']/M['m00'])
	cv2.circle(objects, (cx,cy), 4, (0,0,255), -1)

	print("Area: {}, perimeter: {}".format(area,perimeter))

cv2.imshow("Contours",objects)

cv2.waitKey(0)
cv2.destroyAllWindows()
