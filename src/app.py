##Contador de personas
##Federico Mejia
import numpy as np
import cv2
import Person
import time
import os
import Transform
import Counter
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
from scipy import spatial

if __name__ == '__main__':
    transform = Transform.Transform
    counter = Counter.Counter
    cnt_up   = 0
    cnt_down = 0
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe("../caffe/MobileNetSSD_deploy.prototxt.txt",
                                   "../caffe/MobileNetSSD_deploy.caffemodel")

    #wczytanie filmu
    cap = cv2.VideoCapture('../mov/IMG_1652.MOV')
    # cap = cv2.VideoCapture(0)
    for i in range(19):
        print (i, cap.get(i))
    
    w = cap.get(3)
    h = cap.get(4)
    frameArea = h*w
    areaTH = frameArea/20#frameArea/250
    areaMaxTH = frameArea/2;
    areaMaxWidth = w/2;
    areaMaxHeight = h/2;
    print ('Area Threshold', areaTH)

    # #rysowanie linii
    # line_up = int(1.5*(h/5))
    # line_down   = int(4*(h/5))
    #
    # up_limit =   int(1*(h/5))
    # down_limit = int(4.5*(h/5))
    line_left = int(2 * (w / 5))
    line_right = int(3 * (w / 5))

    up_limit = int(1 * (w / 5))
    down_limit = int(4 * (w / 5))

    print("Red line y:", str(line_right))
    print("Blue line y:", str(line_left))
    line_down_color = (255,0,0)
    line_up_color = (0,0,255)
    pt1 = [line_right, 0];
    pt2 = [line_right, h];
    pts_L1 = np.array([pt1,pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1,1,2))
    pt3 = [line_left, 0];
    pt4 = [line_left, h];
    pts_L2 = np.array([pt3,pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1,1,2))

    pt5 = [up_limit, 0];
    pt6 = [up_limit, h];
    pts_L3 = np.array([pt5,pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1,1,2))
    pt7 = [down_limit, 0];
    pt8 = [down_limit, h];
    pts_L4 = np.array([pt7,pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1,1,2))

    # #background substraction - rozpoznawanie elementow ruszajacych sie
    # fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold = 500, detectShadows = False)
    #
    # #Elementos estructurantes para filtros morfoogicos
    # kernelOp = np.ones((3,3),np.uint8)
    # kernelOp2 = np.ones((5,5),np.uint8)
    # kernelCl = np.ones((11,11),np.uint8)
    
    #Variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    persons = []
    max_p_age = 5
    pid = 1
    img_counter = 0
    person_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        (h1, w1) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.5:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w1, h1, w1, h1])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                for i in persons:
                    i.age_one()  # age every person one frame

                #         fgmask = fgbg.apply(frame)
                #         try:
                #             ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
                #             mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
                #             mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
                #         except:
                #             print('EOF')
                #             print('UP:',cnt_up)
                #             print('DOWN:',cnt_down)
                #             break
                #         #################
                #         #   CONTORNOS   #
                #         #################
                #
                #         # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
                #         _, contours0, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                #         for cnt in contours0:
                #             area = cv2.contourArea(cnt)
                #             x,y,w,h = cv2.boundingRect(cnt)
                #             if area > areaTH and area < areaMaxTH and h<w:
                #################
                #   TRACKING    #
                #################

                M = cv2.moments(box)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                new = True
                if cy in range(up_limit,down_limit):
                    for i in persons:

                        vector = transform.transform(box)
                        persons[i].addVector(vector)


#                         dir_path = "../out/person%s" % person_count;
#                         if not os.path.exists(dir_path):
#                             os.makedirs(dir_path)
#
#                         cv2.imwrite("../out/person%s/img%s.png" % (person_count, img_counter), img)
                        #                         img_counter += 1

                        now = int(time.strftime('%S'))
#                        print(now, " ", i.getLastTime())
                        if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h and abs(now-int(i.getLastTime())) <= 2:

                            new = False
                            i.updateCoords(cx,cy)   #actualiza coordenadas en el objeto and resets age
                            if i.going_UP(line_right, line_left) == True:
                                cnt_up += 1;
                                print ("ID:", i.getId(),'crossed going up at', time.strftime("%c"))
                            elif i.going_DOWN(line_right, line_left) == True:
                                cnt_down += 1;
                                print ("ID:",i.getId(),'crossed going down at',time.strftime("%c"))
                            break
                        if i.getState() == '1':
                            if i.getDir() == 'down' and i.getY() > down_limit:
                                i.setDone()
                            elif i.getDir() == 'up' and i.getY() < up_limit:
                                i.setDone()
                        if i.timedOut():
                            #sacar i de la lista persons
                            index = persons.index(i)
                            persons.pop(index)
                            del i     #liberar la memoria de i

                    if new == True:
                        p = Person.MyPerson(pid,cx,cy, max_p_age)
                        persons.append(p)
                        pid += 1
                        person_count += 1

                    #
                    #
                    #                 #################
                    #                 #   DIBUJOS     #
                    #                 #################
                    #                 cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
                    #                 img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    #                 #cv2.drawContours(frame, cnt, -1, (0,255,0), 3)
                    #
                    #         #END for cnt in contours0
                    #
                    #         #########################
                    #         # DIBUJAR TRAYECTORIAS  #
                    #         #########################
        for i in persons:
    ##        if len(i.getTracks()) >= 2:
    ##            pts = np.array(i.getTracks(), np.int32)
    ##            pts = pts.reshape((-1,1,2))
    ##            frame = cv2.polylines(frame,[pts],False,i.getRGB())
    ##        if i.getId() == 9:
    ##            print str(i.getX()), ',', str(i.getY())
            cv2.putText(frame, str(i.getId()), (i.getX(),i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

        #################
        #   IMAGANES    #
        #################
        str_up = 'UP: '+ str(cnt_up)
        str_down = 'DOWN: '+ str(cnt_down)
        frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
        frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
        frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
        cv2.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, str_down, (10,90), font, 0.5, (255,0,0), 1, cv2.LINE_AA)

        cv2.imshow('Frame',frame)
        #cv2.imshow('Mask',mask)    
        
        #preisonar ESC para salir
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    #END while(cap.isOpened())
        
    #################
    #   LIMPIEZA    #
    #################
    cap.release()
    cv2.destroyAllWindows()