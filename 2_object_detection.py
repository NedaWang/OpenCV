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