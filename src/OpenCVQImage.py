from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import cv2
import threading
import queue

from app import detect


form_class = uic.loadUiType("simple.ui")[0]
raw_frame_queue = queue.Queue()
processed_frames_queue = queue.Queue()
video_width = 1920
video_height = 1080



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
        self.running = threading.Condition()
        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.video_opened = threading.Event()
        self.video_height = None
        self.video_width = None
        self.read_thread = threading.Thread(
            target=video_worker,
            args=(
            0, raw_frame_queue, processed_frames_queue, self.video_width, self.video_height, 30, True, self.start_event,
            self.stop_event))

        self.read_thread.start()


    def start(self):
        self.start_event.set()
        self.stop_event.clear()
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)

    def stop(self):
        self.stop_event.set()
        self.start_event.clear()
        processed_frames_queue.queue.clear()
        raw_frame_queue.queue.clear()
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def update_frame(self):
        if not processed_frames_queue.empty():
            img = processed_frames_queue.get()

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
