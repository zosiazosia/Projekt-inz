from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import cv2
import threading
import queue

import Counter
import Posture
import Transform
import numpy as np
import time

running = False
capture_thread = None
form_class = uic.loadUiType("simple.ui")[0]
frame_queue = queue.Queue()
video_width = 1920
video_height = 1080

counter = Counter.Counter

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
PERSON = 15
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe("../caffe/MobileNetSSD_deploy.prototxt.txt",
                               "../caffe/MobileNetSSD_deploy.caffemodel")


def grab_without_counter(cam, queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    while (running):
        ret, frame = capture.read()

        if queue.qsize() < 10:
            queue.put(frame)
        else:
            print(queue.qsize())

    capture.release()


def grab_with_counter(camera_number, queue, width, height, fps):
    img_counter = 0
    cnt_up = 0
    cnt_down = 0
    pid = 0
    cap = cv2.VideoCapture(camera_number)

    postures = []  # list of active postures
    persons = []
    trans = Transform.Transform(0)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # #rysowanie linii
    line_left = int(2 * (w / 5))
    line_right = int(3 * (w / 5))

    left_limit = int(1 * (w / 5))
    right_limit = int(4 * (w / 5))

    print("Red line y:", str(line_right))
    print("Blue line y:", str(line_left))
    line_down_color = (255, 0, 0)
    line_up_color = (0, 0, 255)
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
    # for each frame
    global running
    while running:
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

                # współrzędne środka masy
                cx = int((endX - startX) / 2 + startX)
                cy = int((endY - startY) / 2 + startY)

                new = True
                # detection between white lines
                if cx in range(left_limit, right_limit):
                    # loop over active postures
                    for p in postures:

                        now = int(time.strftime('%M%S'))
                        if abs(cx - p.getX()) <= 150 and abs(cy - p.getY()) <= 100 and \
                                (abs(now - int(p.getLastTime())) <= 2 or abs(now - int(p.getLastTime())) >= 5958):

                            img = frame[startY:endY, startX:endX]
                            label = "cx%d cy%d time: %d" % (cx, cy, p.getLastTime())
                            cv2.putText(img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            posture_id = '%d' % p.getId()
                            cv2.putText(frame, posture_id, (cx + 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 3, COLORS[idx],
                                        2)

                            posture_id = p.getId()

                            if p.getState() != '1':
                                vector = trans.transform(img)
                                p.addVector(vector)
                            # postures[id].addVector(vector)

                            cv2.imwrite("../out/person%d-%d.png" % (posture_id, img_counter), img)
                            img_counter += 1

                            new = False

                            p.addCoords(cx, cy)
                            if p.going_LEFT(line_left):
                                trans.classify(persons, p, p.getId())

                                cnt_up += 1
                                print("ID:", p.getId(), 'crossed going up at', time.strftime("%c"))

                            elif p.going_RIGHT(line_right):
                                trans.classify(persons, p, p.getId())

                                cnt_down += 1
                                print("ID:", p.getId(), 'crossed going down at', time.strftime("%c"))

                            break

                        # posture outside white space
                        if p.getState() == "1" or p.getX() > right_limit or p.getX() < left_limit:
                            index = postures.index(p)
                            postures.pop(index)
                            del p
                    if new:
                        post = Posture.Posture(pid, cx, cy)
                        postures.append(post)

                        # rebuild tree -> new posture has to be classified
                        if len(persons) != 0:
                            trans.build_tree(persons)

                        # img_counter = 0
                        pid += 1

                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        for i in postures:
            cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

        str_up = 'LEFT: ' + str(cnt_up)
        str_down = 'RIGHT: ' + str(cnt_down)
        frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        if queue.qsize() < 10:
            queue.put(frame)
        else:
            print(queue.qsize())

    cap.release()


class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def set_image(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class MainWindow(QtWidgets.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.startButton.clicked.connect(self.start)
        self.stopButton.clicked.connect(self.stop)
        self.exportButton.setEnabled(False)
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def start(self):
        global running
        running = True
        global capture_thread
        if capture_thread is None or capture_thread._is_stopped:
            capture_thread = threading.Thread(target=grab_with_counter,
                                              args=(1, frame_queue, video_width, video_height, 30))
            capture_thread.start()

        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)

    def stop(self):
        global running
        running = False
        global capture_thread
        capture_thread.join()
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def update_frame(self):
        if not frame_queue.empty():
            img = frame_queue.get()

            # img_height, img_width, img_colors = img.shape
            # scale_w = float(self.window_width) / float(img_width)
            # scale_h = float(self.window_height) / float(img_height)
            # scale = min([scale_w, scale_h])

            # if scale == 0:
            #                scale = 1

            # img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.set_image(image)

    def closeEvent(self, event):
        global running
        running = False


app = QtWidgets.QApplication(sys.argv)
w = MainWindow(None)
w.setWindowTitle('Inteligenty licznik osób')
w.show()
app.exec_()
