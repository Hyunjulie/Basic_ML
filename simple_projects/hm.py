import cv2                              
import numpy as np                           #importing libraries
cap = cv2.VideoCapture(0)                #creating camera objec

while( cap.isOpened()): 
	ret,img = cap.read()                         #reading the frames
	k = cv2.waitKey(10)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	ret,thresh1 = cv2.threshold(blur,150,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	cv2.imshow('input',img)                  #displaying the frames

	if k == 27:
		break

