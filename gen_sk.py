#!/usr/bin/env python

import random
import math

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw

import common
from gen_bg import generate_bg

def euler_to_mat(yaw, pitch, roll):
  # Rotate clockwise about the Y-axis
  c, s = math.cos(yaw), math.sin(yaw)
  M = numpy.matrix([[  c, 0.,  s],
                    [ 0., 1., 0.],
                    [ -s, 0.,  c]])

  # Rotate clockwise about the X-axis
  c, s = math.cos(pitch), math.sin(pitch)
  M = numpy.matrix([[ 1., 0., 0.],
                    [ 0.,  c, -s],
                    [ 0.,  s,  c]]) * M

  # Rotate clockwise about the Z-axis
  c, s = math.cos(roll), math.sin(roll)
  M = numpy.matrix([[  c, -s, 0.],
                    [  s,  c, 0.],
                    [ 0., 0., 1.]]) * M

  return M

def make_affine_transform(from_shape, to_shape,
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
  out_of_bounds = False

  from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
  to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

  scale = random.uniform((min_scale + max_scale) * 0.5 -
                         (max_scale - min_scale) * 0.5 * scale_variation,
                         (min_scale + max_scale) * 0.5 +
                         (max_scale - min_scale) * 0.5 * scale_variation)
  if scale > max_scale or scale < min_scale:
    out_of_bounds = True
  roll = random.uniform(-0.3, 0.3) * rotation_variation
  pitch = random.uniform(-0.2, 0.2) * rotation_variation
  yaw = random.uniform(-1.2, 1.2) * rotation_variation

  # Compute a bounding box on the skewed input image (`from_shape`).
  M = euler_to_mat(yaw, pitch, roll)[:2, :2]
  h, w = from_shape[:2]
  corners = numpy.matrix([[-w, +w, -w, +w],
                          [-h, -h, +h, +h]]) * 0.5
  skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                            numpy.min(M * corners, axis=1))

  # Set the scale as large as possible such that the skewed and scaled shape
  # is less than or equal to the desired ratio in either dimension.
  scale *= numpy.min(to_size / skewed_size)

  # Set the translation such that the skewed and scaled image falls within
  # the output shape's bounds.
  trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
  trans = ((2.0 * trans) ** 5.0) / 2.0
  if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
    out_of_bounds = True
  trans = (to_size - skewed_size * scale) * trans

  center_to = to_size / 2.
  center_from = from_size / 2.

  M = euler_to_mat(yaw, pitch, roll)[:2, :2]
  M *= scale
  M = numpy.hstack([M, trans + center_to - M * center_from])

  return M, out_of_bounds

def generate_plate_number():
  return '{}{}{}{}{}{}{}'.format(
    random.choice(common.LETTERS),
    random.choice(common.LETTERS),
    random.choice(common.DIGITS),
    random.choice(common.DIGITS),
    random.choice(common.LETTERS),
    random.choice(common.LETTERS),
    random.choice(common.LETTERS))

def get_char_image(char):
  file_name = 'np-img/{}.png'.format(char)
  im = cv2.imread(file_name)
  if im is None:
    raise AssertionError('Failed to open file {}'.format(file_name))
  return im

def get_random_plate_template():
  single_row_plate_config = {
    'y': [15, 15, 15, 15, 15, 15, 15],
    'x': [61, 117, 229, 285, 341, 397, 453],
  }

  two_row_plate_config = {
    'y': [11, 11, 105, 105, 105, 105, 105],
    'x': [101, 160, 29, 88, 147, 206, 265],
  }

  file_name = 'np-img/2-row-type-3.jpg'
  im = cv2.imread(file_name)
  if im is None:
    raise AssertionError('Failed to open file {}'.format(file_name))
  return im, two_row_plate_config

  file_name = 'np-img/1-row-type-3.jpg'
  im = cv2.imread(file_name)
  if im is None:
    raise AssertionError('Failed to open file {}'.format(file_name))
  return im, single_row_plate_config

def generate_plate_image():
  plate_number = generate_plate_number()

  pt, ptc = get_random_plate_template()

  print(ptc)

  plate = pt

  for i, char in enumerate(plate_number):
    char_im = get_char_image(char)
    print(i, char_im.shape)
    plate[ptc['y'][i]:ptc['y'][i] + char_im.shape[0], ptc['x'][i]:ptc['x'][i] + char_im.shape[1]] = char_im

  return plate, plate_number

plate_im, plate_number = generate_plate_image()

print(plate_number)

bg = generate_bg()

M, out_of_bounds = make_affine_transform(
                          from_shape=plate_im.shape,
                          to_shape=bg.shape,
                          min_scale=0.6,
                          max_scale=0.875,
                          rotation_variation=1.0,
                          scale_variation=1.1,
                          translation_variation=1.2)

plate_im = cv2.warpAffine(plate_im, M, (bg.shape[1], bg.shape[0]))

cv2.imwrite("plate.jpg", plate_im)
