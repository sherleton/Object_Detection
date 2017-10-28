#import all that is needed
import numpy as np
import cv2

#create an instance for the cascade classifier
#the below xml file is available on github/opencv/data
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#import image
image = cv2.imread('eminem_normal.jpg')
#convert to grayscale
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

#detect the face
faces = face_cascade.detectMultiScale( gray, 1.3, 5)

for (x,y,w,h) in faces:
	#plot the face
	cv2.rectangle(image , (x,y), (x+w, y+h), (255,0,0), 2)

#new image to view
cv2.imwrite("em_face.png" , image)