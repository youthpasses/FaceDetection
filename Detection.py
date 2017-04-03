# coding:utf-8

import tensorflow as tf
import cv2
import os
import sys
import time

checkfaces_path = '/home/makai/Desktop/FaceDetection/Photos/checkfaces/'
FPS = 1

def panda_joke_cnn(X, keep_prob_5, keep_prob_75):
    W_c1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
    b_c1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, W_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob_5)

    W_c2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    b_c2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, W_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob_5)

    W_c3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
    b_c3 = tf.Variable(tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, W_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob_5)

    # Fully connected layer
    W_d = tf.Variable(tf.random_normal([8 * 16 * 32, 512], stddev=0.01))
    b_d = tf.Variable(tf.random_normal([512]))
    dense = tf.reshape(conv3, [-1, W_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, W_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob_75)

    W_out = tf.Variable(tf.random_normal([512, 2], stddev=0.01))
    b_out = tf.Variable(tf.random_normal([2]))
    out = tf.add(tf.matmul(dense, W_out), b_out)
    return out



# saver = tf.train.import_meta_graph('tmp/detectmyface.tfmodel.meta')
# graph = tf.get_default_graph()
# output = graph.get_operation_by_name('output')



X = tf.placeholder(tf.float32, [None, 64, 64, 3])
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

output = panda_joke_cnn(X, keep_prob_5, keep_prob_75)
predict = tf.argmax(output, 1)

saver = tf.train.Saver()
sess =  tf.Session()
saver.restore(sess, "tmp/detectmyface.tfmodel")




def getImageData(imagepath):
    image = cv2.imread(imagepath)
    image_data = image.reshape(1, 64, 64, 3)
    image_data = image_data.astype('float32') / 255.0
    return image_data


def isMyFace(image_data, imagepath):
    res = sess.run(output, feed_dict={X:image_data, keep_prob_5:1, keep_prob_75:1})
    if tf.argmax(res, 1).eval(session=sess)[0] == 1:
        print imagepath, "true"
    else:
        print imagepath, "false"

    # if res[0] == 1:
    #     print "true.."
    # else:
    #     print "false.."


# imagepath = "/home/makai/Desktop/FaceDetection/Photos/myfaces/10.jpg"
# imagepath = "/home/makai/Desktop/FaceDetection/Photos/otherfaces/10.jpg"
# image_data = getImageData(imagepath)
# isMyFace(image_data)




def checkFace():
    face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)
    while True:
        time.sleep(FPS)
        _, img = cam.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_haar.detectMultiScale(gray_img, 1.3, 5)
        if len(faces) > 0:
            face_x,face_y,face_w,face_h = faces[0]
            face = img[face_y:face_y+face_h, face_x:face_x+face_w]
            face = cv2.resize(face, (64, 64))
            imagepath = checkfaces_path + time.strftime('%Y.%m.%d-%H:%M:%S') + '.jpg'
            cv2.imwrite(imagepath, face)
            cv2.imshow('face', face)
            face1 = face.reshape(1,64,64,3)
            face1 = face1.astype('float32') / 255.0
            isMyFace(face1, imagepath)
            
checkFace()