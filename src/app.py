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
from enum import Enum
running = True

# known direction, posture to classify
model_size_x = 224
model_size_y = 224


class direction(Enum):
    LEFT = 'left'
    RIGHT = 'right'

def run_video_counter(cam, queue, gui, layer_name, direction, counter_queue, running_queue):
    trans = Transform.Transform(0, layer_name)
    counter = Counter.Counter(direction)  # or 'right'
    PERSON = 15

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




    # Variables
    font = cv2.FONT_HERSHEY_SIMPLEX

    # list of active postures
    postures = []
    persons = []
    pid = 0
    running = True

    if not gui:
        running = cap.isOpened()

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
                    if not gui:
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
                                if not gui:
                                    cv2.putText(frame, posture_id, (cx + 2, cy),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN_COLOR, 2)

                                # if posture is not yet counted and has not more than 10 vectors
                                if p.getState() != Posture.state.COUNTED and len(p.getVectors()) < 10:
                                    try:
                                        imgT = cv2.resize(img, (model_size_x, model_size_y))
                                        vector = trans.transform(imgT)
                                        p.addVector(vector)
                                    except:
                                        print("")

                                new = False

                                p.addCoords(cx, cy)
                                if p.going_IN(line_left, line_right, counter):
                                    trans.classify(p, counter)
                                    print("ID:", p.getId(), 'crossed going in at', time.strftime("%c"))

                                elif p.going_OUT(line_left, line_right, counter):
                                    trans.classify(p, counter)
                                    print("ID:", p.getId(), 'crossed going out at', time.strftime("%c"))

                                break

                            # posture outside white space
                            if p.getState() == Posture.state.COUNTED or p.getX() > right_limit or p.getX() < left_limit:
                                index = postures.index(p)
                                postures.pop(index)
                                del p
                        if new:
                            post = Posture.Posture(pid, cx, cy)
                            postures.append(post)
                            pid += 1

                        if not gui:
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            if gui:
                counter_queue.put(counter)
            else:
                frame = write_result_on_frame(counter=counter, frame=frame, font=font, line_right=line_right,
                                              line_left=line_left, left_limit=left_limit, right_limit=right_limit, h=h)
                for i in postures:
                    cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

            if gui:
                if queue.qsize() < 10:
                    queue.put(frame)
                else:
                    print(queue.qsize())
            else:
                cv2.imshow('Frame', frame)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            if not cap.isOpened():
                break

            if not running_queue.empty():
                running = running_queue.get()

        else:
            break

    logger = logging.getLogger('recognition')
    logger.setLevel(logging.INFO)
    logger.info(counter.generate_report('eng') + "cnt_in: " + counter.getRegularInString()
                + "cnt_out: " + counter.getRegularOutString())
    cap.release()
    cv2.destroyAllWindows()


def write_result_on_frame(counter, frame, font, line_right, line_left, left_limit, right_limit, h, ):
    RED_COLOR = (255, 0, 0)
    BLUE_COLOR = (0, 0, 255)
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

    frame = cv2.polylines(frame, [pts_L1], False, line_left_color, thickness=2)
    frame = cv2.polylines(frame, [pts_L2], False, line_right_color, thickness=2)
    frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
    frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)

    str_in = 'IN: ' + counter.getRegularInString()
    str_out = 'OUT: ' + counter.getRegularOutString()
    str_rein = 'RE_IN: ' + counter.getReidentInString()
    str_reout = 'RE_OUT: ' + counter.getReidentOutString()
    str_inside = 'INSIDE: ' + counter.getAreInsideString()

    cv2.putText(frame, str_in, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str_out, (10, 70), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str_rein, (10, 100), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str_reout, (10, 130), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str_inside, (10, 160), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return frame


if __name__ == '__main__':
    run_video_counter(cam='../mov/Sekcja_2.mov', queue=queue.Queue(), gui=False,
                      layer_name='block5_conv2', direction=direction.LEFT, counter_queue=queue.Queue(),
                      running_queue=queue.Queue())
