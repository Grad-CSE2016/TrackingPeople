import cv2
import numpy as np
from sort import *



class TrackingPeople:
        def __init__(self):
            super(TrackingPeople, self).__init__()
            mog_er_w = 7
            mog_er_h = 7
            mog_di_w = 16
            mog_di_h = 26
            cv2.ocl.setUseOpenCL(False)
            self.bgs_mog = cv2.createBackgroundSubtractorMOG2()
            self.for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mog_er_w, mog_er_h))
            self.for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mog_di_w, mog_di_h))

            self.mot_tracker = Sort(15)
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        def get_frame(self,image):
            # grey_image = self.bgs_mog.apply(image)
            # thresh, im_bw = cv2.threshold(grey_image, 225, 255, cv2.THRESH_BINARY)
            # im_er = cv2.erode(im_bw, self.for_er)
            # im_dl = cv2.dilate(im_er, self.for_di)
            # _, contours, hierarchy = cv2.findContours(im_dl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # coordinates = []
            # for cnt in contours:
            #     try:
            #         x,y,w,h = cv2.boundingRect(cnt)
            #         #cv2.rectangle(f, (x,y), (x+w, y+h), (255,0,0), 2)
            #         coordinates.append((x,y,w,h))
            #     except:
            #         print ("Bad Rect")
            (rects,weights)=self.hog.detectMultiScale(image,winStride=(8,8),padding=(32,32), scale = 1.05)
            if(len(rects)>0):
                pick=rects[rects[:,3]<250]
                pick = np.array([[x,y,x+w,y+h] for (x,y,w,h) in pick])
                detections=pick
                track_bbs_ids = self.mot_tracker.update(detections)
                return(track_bbs_ids,pick,len(pick))
            else:
                return(np.array([]),np.array([]),0)
