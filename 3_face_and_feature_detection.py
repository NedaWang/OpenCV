# 04_03 template matching
import numpy as np
import cv2

template = cv2.imread('template.jpg',0)  # grayscale
frame = cv2.imread("players.jpg",0)

cv2.imshow("Frame",frame)
cv2.imshow("Template",template)

result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

# getting the maximum brightness out of the result image
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(max_val,max_loc)
# 15: radius, 255: color white, thickness 2
cv2.circle(result,max_loc, 15,255,2)
cv2.imshow("Matching",result)

cv2.waitKey(0)
cv2.destroyAllWindows()
