# coding:utf-8

import cv2
import os
import sys


FACE_ORIGIN_DIR = "/home/makai/Desktop/FaceDetection/Photos/faces/"
FACE_SAVED_DIR = "/home/makai/Desktop/FaceDetection/Photos/otherfaces/"

face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

i = 0
for dirpath, dirnames, filenames in os.walk(FACE_ORIGIN_DIR):
    # print filenames
    for filename in filenames:
        if filename.endswith('.jpg'):
            imagepath = os.path.join(dirpath, filename)
            print "processing with " + imagepath
            img = cv2.imread(imagepath)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_haar.detectMultiScale(gray_img, 1.3, 5)
            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                face = cv2.resize(face, (64, 64))
                cv2.imshow("img", face)
                cv2.imwrite(os.path.join(FACE_SAVED_DIR, str(i) + '.jpg'), face)
                print i
                i+=1
            else:
                print "no face"
