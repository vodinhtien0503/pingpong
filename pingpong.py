import cv2
import numpy as np
import imutils 

cap=cv2.VideoCapture("pingpong.mp4")
while(True):
    ret, frame=cap.read()
    blur=cv2.GaussianBlur(frame,(11,11),0)
    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    # gia tri mau cua qua banh
    lower=np.array([11,109,208])
    upper=np.array([25,256,255])
    mask=cv2.inRange(hsv,lower,upper)
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    # ve dg tron quanh qua banh
    ball_cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ball_cnts=imutils.grab_contours(ball_cnts)
    if len(ball_cnts)>0:
        c=max(ball_cnts,key=cv2.contourArea)
        ((x,y),radius)=cv2.minEnclosingCircle(c)
        if radius>10:
            cv2.circle(frame,(int(x),int(y)),int(radius),(0,0,255),5)
    cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows() 