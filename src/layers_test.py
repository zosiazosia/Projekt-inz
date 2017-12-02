import logging
import queue

from keras.applications import VGG19

from app import run_video_counter

layers = ['input_1', 'block1_conv1', 'block1_conv2', 'block1_pool']

# create logger with 'spam_application'
logger = logging.getLogger('recognition')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('recognition.log')
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

if __name__ == '__main__':
    base_model = VGG19(weights='imagenet')
    for layer in base_model.layers:
        if layer.name != 'input_1':
            run_video_counter(cam='../mov/schody_2.mov', queue=queue.Queue(),
                              width=None, height=None, fps=None, gui=False, layer_name=layer.name, direction='left')
