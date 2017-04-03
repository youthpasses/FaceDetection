# coding:utf-8

import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random
import sys

my_image_path = '/home/makai/Desktop/FaceDetection/Photos/myfaces/'
others_image_path = '/home/makai/Desktop/FaceDetection/Photos/otherfaces/'

image_data = []
label_data = []


def get_padding_size(image):
    h, w, _ = image.shape
    longest_edge = max(h, w)
    top, bottom, left, right = (0, 0, 0, 0)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    return top, bottom, left, right


def read_data(img_path, label):
    for filename in os.listdir(img_path):
        if filename.endswith('.jpg'):
            filepath = os.path.join(img_path, filename)
            image = cv2.imread(filepath)

            top, bottom, left, right = get_padding_size(image)
            image_pad = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image = cv2.resize(image_pad, (64, 64))

            image_data.append(image)
            label_data.append(label)


read_data(others_image_path, '0')
read_data(my_image_path, '1')

image_data = np.array(image_data)
label_data = np.array([[0, 1] if label == '1' else [1, 0] for label in label_data])

train_x, test_x, train_y, test_y = train_test_split(image_data, label_data, test_size=0.05,
                                                    random_state=random.randint(0, 100))

# image (height=64 width=64 channel=3)
train_x = train_x.reshape(train_x.shape[0], 64, 64, 3)
test_x = test_x.reshape(test_x.shape[0], 64, 64, 3)

# nomalize
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

#############################################################
batch_size = 128
num_batch = len(train_x) // batch_size

X = tf.placeholder(tf.float32, [None, 64, 64, 3])  # 图片大小64x64 channel=3
Y = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


def panda_joke_cnn():
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


def train_cnn():
    output = panda_joke_cnn()

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())

        for e in range(50):
            for i in range(num_batch):
                batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                _, loss_, summary = sess.run([optimizer, loss, merged_summary_op],
                                             feed_dict={X: batch_x, Y: batch_y, keep_prob_5: 0.5, keep_prob_75: 0.75})

                summary_writer.add_summary(summary, e * num_batch + i)
                print(e * num_batch + i, loss_)

                if (e * num_batch + i) % 100 == 0:
                    acc = accuracy.eval({X: test_x, Y: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                    print(e * num_batch + i, acc)
                    # save model
                    if acc > 0.98:
                        saver.save(sess, "tmp/detectmyface.tfmodel")
                        sys.exit(0)

































'''
def get_padding_size(image):
    h, w, _ = image.shape
    longest_edge = max(h, w)
    top, bottom, left, right = (0, 0, 0, 0)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    return top, bottom, left, right


def read_data(img_path, label, image_h=64, image_w=64):
    image_data = []
    label_data = []
    for filename in os.listdir(img_path):
        if filename.endswith('.jpg'):
            filepath = os.path.join(img_path, filename)
            image = cv2.imread(filepath)
            top, bottom, left, right = get_padding_size(image)
            image_pad = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image = cv2.resize(image_pad, (image_h, image_w))
            image_data.append(image)
            label_data.append(label)
    return image_data, label_data


def getData():
    image0, label0 = read_data(others_image_path, 0)
    image1, label1 = read_data(my_image_path, 1)
    image0.extend(image1)
    label0.extend(label1)
    image0 = np.array(image0)
    label0 = np.array(label0)
    count = len(label0)
    label = np.zeros((count, 2))
    label[np.arange(count), label0] = 1;
    image0 = image0.reshape(image0.shape[0], 64, 64, 3)
    image0 = image0.astype('float32') / 255.0
    return image0, label



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


def train_cnn():

    # 获取数据
    image_data, image_label = getData()
    train_x, test_x, train_y, test_y = train_test_split(image_data, image_label, test_size=0.05)
    train_x, vali_x, train_y, vali_y = train_test_split(train_x, train_y, test_size=0.05)

    print("train: ", train_x.shape, train_y.shape)
    print("vali: ", vali_x.shape, vali_y.shape)
    print("test: ", test_x.shape, test_y.shape)
    print train_x[0]

    # image (height=64 width=64 channel=3)
    # train_x = train_x.reshape(train_x.shape[0], 64, 64, 3)
    # test_x = test_x.reshape(test_x.shape[0], 64, 64, 3)
    # # nomalize
    # train_x = train_x.astype('float32') / 255.0
    # test_x = test_x.astype('float32') / 255.0


    #############################################################


    X = tf.placeholder(tf.float32, [None, 64, 64, 3])  # 图片大小64x64 channel=3
    Y = tf.placeholder(tf.float32, [None, 2])

    keep_prob_5 = tf.placeholder(tf.float32)
    keep_prob_75 = tf.placeholder(tf.float32)

    # 获取模型
    output = panda_joke_cnn(X, keep_prob_5, keep_prob_75)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    batch_size = 128
    num_batch = len(train_x) // batch_size
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())

        for e in range(5):
            for i in range(num_batch):
                batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                _, loss_, summary = sess.run([optimizer, loss, merged_summary_op],
                                             feed_dict={X: batch_x, Y: batch_y, keep_prob_5: 0.5, keep_prob_75: 0.75})

                summary_writer.add_summary(summary, e * num_batch + i)
                #print(e * num_batch + i, loss_)

                if i % 10 == 0:
                    # acc = accuracy.eval({X: vali_x, Y: vali_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                    # cost, acc_train = sess.run([loss, accuracy], feed_dict={X: train_x, Y: train_y, keep_prob_5:1, keep_prob_75:1});
                    # acc_vali = sess.run(accuracy, feed_dict={X: vali_x, Y: vali_y, keep_prob_5: 1, keep_prob_75: 1});
                    print 'Iter: ' + str(e * num_batch + i), \
                        ', Train loss: %g'%loss.eval(feed_dict={X:train_x, Y:train_y, keep_prob_5:1, keep_prob_75:1}),\
                        ', Train acc: %g'%accuracy.eval(feed_dict={X:train_x, Y:train_y, keep_prob_5:1, keep_prob_75:1}),\
                        ', Vali acc: %g'%accuracy.eval(feed_dict={X:vali_x, Y:vali_y, keep_prob_5:1, keep_prob_75:1})
        print "Optimization Finished!"
        print "Test acc: ", sess.run(accuracy, feed_dict={X:test_x, Y:test_y})
        # save model
        saver.save(sess, "tmp/detection.model", global_step=e * num_batch + i)
        sys.exit(0)

'''
train_cnn()