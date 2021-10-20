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




# 04_05 face detection
img = cv2.imread("faces.jpeg",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
path = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(path)

#scaleFactor: compensating factor for only wanting faces close to the camera
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(40,40))
print(len(faces))

# x, y within height variable
for (x, y, w, h) in faces:
	# (x, y): top left corner
	# (x+w,y+h): bottom right corner
	cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
