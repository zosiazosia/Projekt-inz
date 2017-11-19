import numpy as np
import cv2
import Person
import time
import os
import Transform
import Counter
import Posture

# known direction, posture to classify


if __name__ == '__main__':
    trans = Transform.Transform(0)
    counter = Counter.Counter
    cnt_up   = 0
    cnt_down = 0
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    PERSON = 15
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe("../caffe/MobileNetSSD_deploy.prototxt.txt",
                                   "../caffe/MobileNetSSD_deploy.caffemodel")

    #wczytanie filmu
    # cap = cv2.VideoCapture('../mov/schody_2.mov')
    #    cap = cv2.VideoCapture('../mov/IMG_1652.MOV')
    cap = cv2.VideoCapture(1)
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
    line_left = int(2 * (w / 5))
    line_right = int(3 * (w / 5))

    left_limit = int(1 * (w / 5))
    right_limit = int(4 * (w / 5))

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

    pt5 = [left_limit, 0];
    pt6 = [left_limit, h];
    pts_L3 = np.array([pt5,pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1,1,2))
    pt7 = [right_limit, 0];
    pt8 = [right_limit, h];
    pts_L4 = np.array([pt7,pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1,1,2))

    #Variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    postures = []  #list of active postures
    persons = []
    pid = 0
    img_counter = 0
    frame_num = 0

    #for each frame
    while cap.isOpened():
        ret, frame = cap.read()

        # co druga ramka
        # frame_num += 1
        # if frame_num % 2:
        #     continue;
        (h1, w1) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            idx = int(detections[0, 0, i, 1])  # idx = 15 dla person
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.3 and idx == PERSON:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w1, h1, w1, h1])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                #współrzędne środka masy
                cx = int((endX - startX) / 2 + startX)
                cy = int((endY - startY) / 2 + startY)

                new = True
                # detection between white lines
                if cx in range(left_limit, right_limit):
                    # loop over active postures
                    for p in postures:

                        now = int(time.strftime('%M%S'))
                        if abs(cx - p.getX()) <= 150 and abs(cy - p.getY()) <= 100 and (
                                        abs(now - int(p.getLastTime())) <= 2 or abs(
                                        now - int(p.getLastTime())) >= 5958):

                            img = frame[startY:endY, startX:endX]
                            label = "cx%d cy%d time: %d" % (cx, cy, p.getLastTime())
                            cv2.putText(img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            id = '%d' % p.getId()
                            cv2.putText(frame, id, (cx + 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 3, COLORS[idx], 2)

                            id = p.getId()

                            if p.getState() != '1':
                                vector = trans.transform(img)
                                p.addVector(vector)
                            #postures[id].addVector(vector)

                            cv2.imwrite("../out/person%d-%d.png" % (id, img_counter), img)
                            img_counter += 1

                            new = False

                            p.addCoords(cx, cy)
                            if p.going_LEFT(line_left) == True:
                                trans.classify(persons, p, p.getId())

                                cnt_up += 1;
                                print("ID:", p.getId(), 'crossed going up at', time.strftime("%c"))

                            elif p.going_RIGHT(line_right) == True:
                                trans.classify(persons, p, p.getId())

                                cnt_down += 1;
                                print("ID:", p.getId(), 'crossed going down at', time.strftime("%c"))

                            break
                        # if p.getState() == '1':
                        #     if p.getDir() == 'right' and p.getX() > right_limit:
                        #         p.setDone()
                        #     elif p.getDir() == 'left' and p.getX() < left_limit:
                        #         p.setDone()

                        # posture outside white space
                        if (p.getState() == "1" or p.getX() > right_limit or p.getX() < left_limit):
                            index = postures.index(p)
                            postures.pop(index)
                            del p
                    if new == True:
                        post = Posture.Posture(pid, cx, cy)
                        postures.append(post)

                        # rebuild tree -> new posture has to be classified
                        if (len(persons) != 0):
                            trans.build_tree(persons)


                        # img_counter = 0
                        pid += 1

                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        for i in postures:
            cv2.putText(frame, str(i.getId()), (i.getX(),i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

        str_up = 'LEFT: ' + str(cnt_up)
        str_down = 'RIGHT: ' + str(cnt_down)
        frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
        frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
        frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
        cv2.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, str_down, (10,90), font, 0.5, (255,0,0), 1, cv2.LINE_AA)

        cv2.imshow('Frame',frame)
        
        #preisonar ESC para salir
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()