from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import cv2
import threading
import queue

from app import run_video_counter

running = False
capture_thread = None
form_class = uic.loadUiType("simple.ui")[0]
frame_queue = queue.Queue()
video_width = 1920
video_height = 1080


def grab_without_counter(cam, queue, width, height, fps, gui):
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
            capture_thread = threading.Thread(target=run_video_counter,
                                              args=(1, frame_queue, video_width, video_height, 30, True))
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
