from image_tools import *
from detection import *

import pygame as pg
import cv2
import os


base_dir = os.path.dirname(__file__) + '/..'
img = cv2.imread(base_dir + '/test/moon.jpg', flags=cv2.CV_8UC4)
img_d = cv2.cvtColor(cv2.imread(base_dir + '/test/moon_depth_map.png'), cv2.COLOR_RGB2GRAY)
img = cv2.resize(img, img_d.shape[:2])

scl_fct = 1
off_set = 20


layers = get_layers(img, img_d)

win = pg.display.set_mode((int(img_d.shape[0]*scl_fct - off_set*scl_fct), int(img_d.shape[1]*scl_fct - off_set*scl_fct)))

px = 0
py = 0
x_transform = True
y_transform = False
run = True

cap = cv2.VideoCapture(0)


while run:

    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    initial_pos = (frame.shape[0]/2, frame.shape[1]/2)

    face_rect = get_face_rect(frame)
    result = frame

    if len(face_rect) != 0:
        x,y,w,h, = face_rect

        px = (initial_pos[0] - (x + w/2))/(50*scl_fct)
        py = (initial_pos[1] - (y + h/2))/(50*scl_fct)

    for event in pg.event.get():
        if event.type==pg.QUIT:
            run = False

    win.fill((255, 255, 255))
    
    for i, layer in enumerate(layers):
        new_x = -off_set/2
        new_y = -off_set/2
        if x_transform:
            new_x = 0 + px*i
        if y_transform:
            new_y = 0 + py*i
        win.blit(pg.transform.scale(conv_cv_pygame_alpha(layer), (int(layer.shape[0]*scl_fct), int(layer.shape[1]*scl_fct))), (new_x,new_y))
    pg.display.update()

cap.release()
cv2.destroyAllWindows()
pg.quit()