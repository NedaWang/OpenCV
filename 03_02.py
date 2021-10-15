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
