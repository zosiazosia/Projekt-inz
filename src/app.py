import logging
import queue

import numpy as np
import cv2
import Person
import time
import os
import Transform
import Counter
import Posture
import sys

running = True

# known direction, posture to classify
model_size_x = 224
model_size_y = 224

def run_video_counter(cam, queue, width, height, fps, gui, layer_name):
    trans = Transform.Transform(0, layer_name)
    counter = Counter.Counter('left')  # or 'right'
    cnt_left = 0
    cnt_right = 0
    PERSON = 15
    PERSON_STRING = "Person"
    RED_COLOR = (255, 0, 0)
    BLUE_COLOR = (0, 0, 255)
    GREEN_COLOR = (0, 255, 0)
    net = cv2.dnn.readNetFromCaffe("../caffe/MobileNetSSD_deploy.prototxt.txt",
                                   "../caffe/MobileNetSSD_deploy.caffemodel")

    # wczytanie filmu
    cap = cv2.VideoCapture(cam)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # limit lines
    line_left = int(2 * (w / 5))
    line_right = int(3 * (w / 5))

    left_limit = int(1 * (w / 5))
    right_limit = int(4 * (w / 5))

    line_left_color = RED_COLOR
    line_right_color = BLUE_COLOR
    pt1 = [line_right, 0]
    pt2 = [line_right, h]
    pts_L1 = np.array([pt1, pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1, 1, 2))
    pt3 = [line_left, 0]
    pt4 = [line_left, h]
    pts_L2 = np.array([pt3, pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1, 1, 2))

    pt5 = [left_limit, 0]
    pt6 = [left_limit, h]
    pts_L3 = np.array([pt5, pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1, 1, 2))
    pt7 = [right_limit, 0]
    pt8 = [right_limit, h]
    pts_L4 = np.array([pt7, pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1, 1, 2))

    # Variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    postures = []  # list of active postures
    persons = []
    pid = 0
    global running
    if not gui:
        running = cap.isOpened()

    start_time = time.time()
    end_time = 0
    frame_quantity = 0

    # for each frame
    while running:
        ret, frame = cap.read()
        if ret:
            # co druga ramka
            #  frame_num += 1
            #  if frame_num % 2:
            #      continue

            # save height and width of a frame
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
                    # label = "{}: {:.2f}%".format(PERSON_STRING, confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), GREEN_COLOR, 2)
                    # y = startY - 15 if startY - 15 > 15 else startY + 15
                    # cv2.putText(frame, label, (startX, y),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN_COLOR, 2)

                    # współrzędne środka prostokąta
                    cx = int((endX - startX) / 2 + startX)
                    cy = int((endY - startY) / 2 + startY)

                    new = True
                    # detection between white lines
                    if cx in range(left_limit, right_limit):
                        # loop over active postures
                        for p in postures:

                            now = int(time.strftime('%M%S'))

                            if abs(cx - p.getX()) <= w / 5 and abs(cy - p.getY()) <= h / 4 \
                                    and (abs(now - int(p.getLastTime())) <= 4 or abs(
                                            now - int(p.getLastTime())) >= 5955):

                                img = frame[startY:endY, startX:endX]
                                # label = "cx%d cy%d time: %d" % (cx, cy, p.getLastTime())
                                # cv2.putText(img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN_COLOR, 2)
                                posture_id = '%d' % p.getId()
                                cv2.putText(frame, posture_id, (cx + 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN_COLOR,
                                            2)

                                # if posture is not yet counted and has not more than 10 vectors
                                if p.getState() != '1' and len(p.getVectors()) < 10:
                                    try:
                                        imgT = cv2.resize(img, (model_size_x, model_size_y))
                                        vector = trans.transform(imgT)
                                        p.addVector(vector)
                                    except:
                                        print("Unexpected error:", sys.exc_info()[0])

                                # cv2.imwrite("../out/person%d-%d.png" % (posture_id, img_counter), img)

                                new = False

                                p.addCoords(cx, cy)
                                if p.going_IN(line_left, line_right, counter):
                                    trans.classify(p, counter)

                                    cnt_left += 1
                                    print("ID:", p.getId(), 'crossed going left at', time.strftime("%c"))

                                elif p.going_OUT(line_left, line_right, counter):
                                    trans.classify(p, counter)

                                    cnt_right += 1
                                    print("ID:", p.getId(), 'crossed going right at', time.strftime("%c"))

                                break
                            # if p.getState() == '1':
                            #     if p.getDir() == 'right' and p.getX() > right_limit:
                            #         p.setDone()
                            #     elif p.getDir() == 'left' and p.getX() < left_limit:
                            #         p.setDone()

                            # posture outside white space
                            if p.getState() == "1" or p.getX() > right_limit or p.getX() < left_limit:
                                index = postures.index(p)
                                postures.pop(index)
                                del p
                        if new:
                            post = Posture.Posture(pid, cx, cy)
                            postures.append(post)

                            # rebuild tree -> new posture has to be classified
                            # if len(persons) != 0:
                            #     trans.build_tree(persons)
                            pid += 1

                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            for i in postures:
                cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

            str_up = 'LEFT: ' + str(cnt_left)
            str_down = 'RIGHT: ' + str(cnt_right)
            str_in = 'IN: ' + str(counter.getCameIn())
            str_out = 'OUT: ' + str(counter.getCameOut())
            str_rein = 'RE_IN: ' + str(counter.getReidentIn())
            str_reout = 'RE_OUT: ' + str(counter.getReidentOut())
            str_inside = 'INSIDE: ' + str(counter.getAreInside())
            frame = cv2.polylines(frame, [pts_L1], False, line_left_color, thickness=2)
            frame = cv2.polylines(frame, [pts_L2], False, line_right_color, thickness=2)
            frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
            frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
            cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str_down, (10, 70), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str_down, (10, 70), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str_in, (10, 100), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str_out, (10, 130), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str_rein, (10, 160), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str_reout, (10, 190), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str_inside, (10, 220), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            if gui:
                if queue.qsize() < 10:
                    queue.put(frame)
                    frame_quantity += 1
                else:
                    print("app line 211: " + str(queue.qsize()))
            else:
                cv2.imshow('Frame', frame)
                frame_quantity += 1
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    end_time = time.time()
                    break
            if not cap.isOpened():
                end_time = time.time()
                break
        else:
            break

    time_elapsed = end_time - start_time
    logger = logging.getLogger('recognition')
    logger.info("Program has finished. Time " + str(time_elapsed) + "  frames:  " + str(frame_quantity))
    ftp = frame_quantity / time_elapsed
    logger.info("FPT: " + str(ftp))
    logger.setLevel(logging.INFO)
    logger.info(counter.generate_report() + "cnt_left: " + str(cnt_left) + "cnt_right: " + str(cnt_right))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_video_counter(cam='../mov/Sekcja_2.mov', queue=queue.Queue(), width=None, height=None, fps=None, gui=False,
                      layer_name='block4_pool')
