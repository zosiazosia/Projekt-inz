import logging

import cv2
import time

logger = logging.getLogger('camera_test')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('../logs/camera_test.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

start_time = time.time()
end_time = 0
frame_quantity = 0

import numpy as np
import cv2

cap = cv2.VideoCapture(1)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    frame_quantity += 1
    if frame_quantity == 909:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
end_time = time.time()
time_elapsed = end_time - start_time
logger.info("Program has finished. Time " + str(time_elapsed) + "  frames:  " + str(frame_quantity))
ftp = frame_quantity / time_elapsed
logger.info("FPT: " + str(ftp))
logger.setLevel(logging.INFO)
cap.release()
cv2.destroyAllWindows()
