import os, sys

sys.path.insert(1, os.path.dirname(__file__)+'/../MiDaS')

import midas

from PIL import Image
from PIL import ImageDraw
import numpy as np
import pygame as pg
import cv2



def get_rects(mask):
    mask_cv = np.asarray(mask) 
    contours, hierarchy = cv2.findContours(mask_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    for contour in contours:
        rect = cv2.boundingRect(contour)
        rects.append(rect)

    return rects


def draw_rects(img, rects):

    img_draw = ImageDraw.Draw(img)

    for rect in rects:
        x, y, w, h = rect 
        shape = [(x, y), (x + w, y + h)]
        img_draw.rectangle(shape, outline=(0, 255, 0))

    img.show()



def conv_cv_pygame(cv_image, mode='RGB'):
    size = cv_image.shape[1::-1]
    data = cv_image.tobytes()
     
    frame_pg = pg.image.fromstring(data, size, mode)
          
    return frame_pg



def conv_cv_alpha(cv_image, mask):
    b, g, r = cv2.split(cv_image)
    rgba = [r, g, b, mask]
    cv_image = cv2.merge(rgba,4)
          
    return cv_image


def conv_pil_pygame(cv_image):
    mode = cv_image.mode
    size = cv_image.size
    data = cv_image.tobytes()
        
    frame_pg = pg.image.fromstring(data, size, mode)
          
    return frame_pg

def conv_pil_cv(pil_image):
    cv_image = np.array(pil_image) 
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    return cv_image


def get_depth(img):
    depth_map = midas.run(img, model_type='small', optimize=False)

    return depth_map


def get_layers(img, depth_map, div=30):
    layers = []

    prev_thres = 255

    for thres in range(255 - div, 0, -div):
        ret, mask = cv2.threshold(depth_map, thres, 255, cv2.THRESH_BINARY)
        ret, prev_mask = cv2.threshold(depth_map, prev_thres, 255, cv2.THRESH_BINARY)

        prev_thres = thres

        inpaint_img = cv2.inpaint(img, prev_mask, 10, cv2.INPAINT_NS)
        layer = cv2.bitwise_and(inpaint_img, inpaint_img, mask = mask)

        layers.append(conv_cv_alpha(layer, mask))


    # adding last layer
    mask = np.zeros(depth_map.shape, np.uint8)
    mask[:,:] = 255
    ret, prev_mask = cv2.threshold(depth_map, prev_thres, 255, cv2.THRESH_BINARY)

    inpaint_img = cv2.inpaint(img, prev_mask, 10, cv2.INPAINT_NS)
    layer = cv2.bitwise_and(inpaint_img, inpaint_img, mask = mask)

    layers.append(conv_cv_alpha(layer, mask))

    return layers[::-1]