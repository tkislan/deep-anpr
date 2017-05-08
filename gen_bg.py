import random

import cv2

OUTPUT_SHAPE = (128, 256)

def generate_bg():
  im = get_bg()

  x = random.randint(0, im.shape[1] - OUTPUT_SHAPE[1])
  y = random.randint(0, im.shape[0] - OUTPUT_SHAPE[0])
  im = im[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

  return im

def get_bg():
  file_name = "np-img/streetview.png"
  im = cv2.imread(file_name)

  if im is None:
    raise AssertionError('Failed to open file {}'.format(file_name))

  return im
