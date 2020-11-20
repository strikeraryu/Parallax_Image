import os

import cv2
import numpy


base_dir = os.path.dirname(__file__) + '/..'
face_cascade = cv2.CascadeClassifier(base_dir + '/classifier/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(base_dir + '/classifier/haarcascade_eye.xml')   


def get_face_rect(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    if len(face_rects) == 0: 
        return ()

    return face_rects[0]

