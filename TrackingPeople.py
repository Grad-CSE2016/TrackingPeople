import cv2
import numpy as np
from sort import *
import glob
import time


class TrackingPeople:
        def __init__(self):
            super(TrackingPeople, self).__init__()

            self.mot_tracker = Sort(15,2)
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.peopleCount=0
            self.lastCounted=0
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
                for(trk)in track_bbs_ids:
                    if(trk[4]>self.lastCounted):
                        self.lastCounted=trk[4]
                        self.peopleCount=self.peopleCount+1
                return(track_bbs_ids,pick,self.peopleCount)
            else:
                return(np.array([]),np.array([]),self.peopleCount)

if __name__ == '__main__':
    # """# Run a demo.# """
    cv2.namedWindow("tracking")
    camera = cv2.VideoCapture("/home/ahmed/Desktop/dataset/6.avi")

    RedFram=20
    font = cv2.FONT_HERSHEY_SIMPLEX
    Track=TrackingPeople()
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32,3)
    while camera.isOpened():
        ok, image=camera.read()

        print(total_frames)
        print(total_time)
        total_frames=total_frames+1
        start_time = time.time()

        trackers,detections,peopleCount=Track.get_frame(image)
        for(tracker)in trackers:
            x1,y1,x2,y2,n=int(tracker[0]),int(tracker[1]),int(tracker[2]),int(tracker[3]),int(tracker[4]),;
            color=colours[n%32,:]
            (R,G,B)=int(color[0]*255),int(color[1]*255),int(color[2]*255)
            cv2.putText(image,str(n),(x1,y1), font, 1, (B,G,R), 2, cv2.LINE_AA)
            cv2.rectangle(image, (x1+RedFram,y1+RedFram), (x2-RedFram, y2-RedFram), (B,G,R), 2)
        cv2.putText(image,str(peopleCount),(0,50), font, 2, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("tracking", image)

        cycle_time = time.time() - start_time
        total_time += cycle_time

        k = cv2.waitKey(1) & 0xff
        if k == 27 : break # esc pressed
    camera.release()
    out.release()
    cv2.destroyAllWindows()
