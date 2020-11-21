import os
import tkinter
from tkinter import filedialog

import pygame as pg
import cv2

from image_utils import *
from detection import *


base_dir = os.path.dirname(__file__) + '/..'


def create_parallax_image(img_path=None, height=-1, width=-1, layer_div = 30, show_dmap=False):
    if img_path == None:
        tkinter.Tk().withdraw()
        img_path = filedialog.askopenfilename(initialdir = base_dir + '/test')

    img = cv2.imread(img_path)

    if height==-1 or width==-1:
        try:
            img_ratio  = img.shape[0]/img.shape[1]
        except Exception as e:
            print('No image')
            return []
        if height==-1 and width==-1:
            width, height = int(800/img_ratio), 800
        elif height==-1:
            width, height = width, width*img_ratio
        else:
            width, height = height/img_ratio, height


    print('\nCreating depth map...')
    img = cv2.resize(img, (width, height))
    img_depth_map = get_depth(img)
    print('\ndepth map genrated...')

    if show_dmap:
        cv2.imshow('Depth map', img_depth_map)
        cv2.waitKey(0)

    print('\nCreating layers...')
    layers = get_layers(img, img_depth_map, div=layer_div)
    new_layers = []

    print('\nConverting layers..')
    for layer in layers:
        new_layers.append(conv_cv_pygame(layer, mode='RGBA'))

    return new_layers


def show_parallax_image(layers, scale = 1, off_set = 20, x_transform = True, y_transform = False, sens = 50, show_cam = False):

    if len(layers) == 0:
        print('No layers to show')
        return 

    width, height = layers[0].get_width(), layers[0].get_height()    
    win = pg.display.set_mode((int((width - off_set)*scale), int((height - off_set)*scale)))
    pg.display.set_caption('Parallax_image')

    scaled_layers = []
    for layer in layers:
        scaled_layers.append(pg.transform.scale(layer, (int(width*scale), int(height*scale))))

    shift_x = 0
    shift_y = 0
    run = True

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


    while run:

        for event in pg.event.get():
            if event.type==pg.QUIT:
                run = False

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        initial_pos = (frame.shape[0]/2, frame.shape[1]/2)

        face_rect = get_face_rect(frame)

        if len(face_rect) != 0:
            x,y,w,h, = face_rect
            face_rect_frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,0), 3)
            shift_x = (initial_pos[0] - (x + w/2))/(sens*scale)
            shift_y = (initial_pos[1] - (y + h/2))/(sens*scale)


        win.fill((255, 255, 255))

        
        for i, layer in enumerate(scaled_layers):
            new_x = -off_set/2
            new_y = -off_set/2
            if x_transform:
                new_x = 0 + shift_x*i
            if y_transform:
                new_y = 0 + shift_y*i
            win.blit(layer, (new_x, new_y))

        face_rect_frame = cv2.resize(face_rect_frame, (100, 100))
        if show_cam:
            win.blit(conv_cv_pygame(face_rect_frame), (0, 0))
        
        pg.display.update()

    cap.release()
    cv2.destroyAllWindows()
    pg.quit()
