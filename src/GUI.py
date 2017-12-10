import os
import queue
import sys
import threading

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets, uic

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
        self.exportButton.clicked.connect(self.export_report)
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
        self.counting_direction.enabled = True
        self.counter_state = None

    def start(self):
        self.running = True
        self.set_counting_direction()
        if self.capture_thread is None or self.capture_thread._is_stopped:
            self.capture_thread = None
            self.capture_thread = threading.Thread(target=run_video_counter,
                                              args=(
                                              '../mov/Sekcja_2.mov', frame_queue, video_width, video_height, 30, True,
                                              'block4_pool'))

            self.capture_thread.start()

        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.exportButton.setEnabled(False)
        self.counting_direction.setEnabled(False)


    def stop(self):
        self.running_queue.put(False)
        self.running = False
        self.capture_thread.join()
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.frame_queue.queue.clear()
        self.counting_direction.setEnabled(True)
        self.exportButton.setEnabled(True)

    def update_frame(self):
        if not self.frame_queue.empty():
            img = self.frame_queue.get()
            if not self.running:
                grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = grey_img

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.set_image(image)

        if not self.counter_queue.empty():
            self.counter_state = self.counter_queue.get()
            self.counted_in.setText(self.counter_state.getRegularLeftString())
            self.counted_out.setText(self.counter_state.getRegularRightString())
            self.reident_in.setText(self.counter_state.getReidentInString())
            self.reident_out.setText(self.counter_state.getReidentOutString())
            self.inside.setText(self.counter_state.getAreInsideString())

    def export_report(self):
        filename = '../reports/report.txt'

        if not os.path.exists('../reports'):
            os.makedirs('../reports')
        report_content = self.counter_state.generate_report()

        with open(filename, "w+") as f:
            f.write(report_content)
            f.close()

        self.exportButton.setEnabled(False)

    def closeEvent(self, event):
        self.running_queue.put(False)
        self.running = False

    def set_counting_direction(self):
        self.count_direction = 'left'
        if self.going_left.isChecked():
            self.count_direction = 'left'
        if self.going_right.isChecked():
            self.count_direction = 'right'




app = QtWidgets.QApplication(sys.argv)
w = MainWindow(None)
w.setWindowTitle('Inteligentny licznik osób')
w.show()
app.exec_()
