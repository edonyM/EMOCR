import sys
import numpy as np
import cv2

im = cv2.imread('pitrain.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

contours_,hierarchy_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
[_,_,_,h] = cv2.boundingRect(contours_[0])
h_min = h;
for cnt in contours_:
    #if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if h < h_min: h_min = h
#################      Now finding Contours         ###################
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(im,contours,-1,(0,0,255),3)
#cv2.imshow('in', im)
#key = cv2.waitKey(0)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

#nn = []
#for cnt in contours:
#    #if cv2.contourArea(cnt)>50:
#        [x,y,w,h] = cv2.boundingRect(cnt)
#        nn.append(h)
#nn = sorted(nn)
for cnt in contours:
    #if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>=1:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
