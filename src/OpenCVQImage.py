from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import cv2
import threading
import queue

from app import run_video_counter

form_class = uic.loadUiType("simple.ui")[0]

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
        self.count_direction = 'left'
        self.counter_queue = queue.Queue()
        self.running_queue = queue.Queue()
        self.running = False
        self.capture_thread = None
        self.frame_queue = queue.Queue()

    def start(self):
        self.running = True
        if self.capture_thread is None or self.capture_thread._is_stopped:
            self.capture_thread = None
            self.capture_thread = threading.Thread(target=run_video_counter,
                                              args=(
                                                  '../mov/Sekcja_2.mov', self.frame_queue, True,
                                                  'block4_pool', self.count_direction, self.counter_queue,
                                                  self.running_queue))

            self.capture_thread.start()
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)

    def stop(self):
        self.running_queue.put(False)
        self.running = False
        self.capture_thread.join()
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.frame_queue.queue.clear()

    def update_frame(self):
        if not self.frame_queue.empty():
            img = self.frame_queue.get()
            if not self.running:
                grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = grey_img
            # # img_height, img_width, img_colors = img.shape
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

        if not self.counter_queue.empty():
            counter = self.counter_queue.get()
            self.counted_in.setText(str(counter.regular_left))
            self.counted_out.setText(str(counter.regular_right))
            self.reident_in.setText(str(counter.reident_in))
            self.reident_out.setText(str(counter.reident_out))
            self.inside.setText(str(counter.are_inside))


    def closeEvent(self, event):
        self.running_queue.put(False)
        self.running = False


app = QtWidgets.QApplication(sys.argv)
w = MainWindow(None)
w.setWindowTitle('Intelligent people counter')
w.show()
app.exec_()
