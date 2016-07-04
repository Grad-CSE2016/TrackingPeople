import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from imutils import paths
from sort import *

cv2.namedWindow("tracking")
camera = cv2.VideoCapture("/home/ahmed/Desktop/dataset/2.mp4")
# seq_dets = np.loadtxt('det.txt',delimiter=',')  #load detections
# frame=0;
font = cv2.FONT_HERSHEY_SIMPLEX
mot_tracker = Sort()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
colours = np.random.rand(32,3)
while camera.isOpened():
    ok, image=camera.read()

    (rects,weights)=hog.detectMultiScale(image,winStride=(8,8),padding=(32,32), scale = 1.05)
    if(len(rects)>0):
        pick=rects[rects[:,3]<200]
        for (x,y,w,h) in pick:
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 1)
        #print(pick)
        pick = np.array([[x,y,x+w,y+h] for (x,y,w,h) in pick])
        detections=pick
        # detections[:,2:4] += detections[:,0:2]
        track_bbs_ids = mot_tracker.update(detections)
        # print(track_bbs_ids)
        for(tracker)in track_bbs_ids:
            x1,y1,x2,y2,n=int(tracker[0]),int(tracker[1]),int(tracker[2]),int(tracker[3]),int(tracker[4]),;
            #print(tracker)
            color=colours[n%32,:]
            B=int(color[0]*255)
            R=int(color[1]*255)
            G=int(color[2]*255)
            cv2.putText(image,str(n),(x1,y1), font, 1, (R,G,B), 2, cv2.LINE_AA)
            cv2.rectangle(image, (x1,y1), (x2, y2), (R,G,B), 2)
        cv2.putText(image,str(len(pick)),(0,50), font, 2, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("tracking", image)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break # esc pressed
