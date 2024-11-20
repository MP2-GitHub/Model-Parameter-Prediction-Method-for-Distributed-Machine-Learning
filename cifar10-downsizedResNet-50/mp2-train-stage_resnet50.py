"""
code: MP2-train-prediction-model
work-flow:
    1、run mp2-train-stage.py python file.

"""

import time
import tensorflow as tf
import pickle as pk
import numpy as np
from input_data import Cifar10
import math
from mixup import mixup_data
from seblock import SE_block
import random
import utils

# 正则化系数
REGULARIZERATION = 0.0002


def model_parameters_dataset(batch_size, index):

    with open('./model_para_23/xs_layers_123_23_{}'.format(index), 'rb') as f:
        data = pk.load(f)

    train_data = data
    xs = []
    m = train_data.shape[0]
    nums_batches = m // batch_size
    for k in range(0, nums_batches):
        train_xs = train_data[batch_size * k: batch_size * (k + 1)]
        xs.append(train_xs)

    print("Complete dataset loading! The shape of dataset is {}. Start the \"Predict Network\" training!"
          .format(data.shape))

    return xs, nums_batches


def model_parameters_dataset2(batch_size):
    with open('./datasets/train_xs_10layers', 'rb') as f:
        input = pk.load(f)

    # uint-2
    # with open('./datasets/y6', 'rb') as f:
    #     y6 = pk.load(f)
    # with open('./datasets/y7', 'rb') as f:
    #     y7 = pk.load(f)
    # with open('./datasets/y8', 'rb') as f:
    #     y8 = pk.load(f)
    # with open('./datasets/y9', 'rb') as f:
    #     y9 = pk.load(f)
    # with open('./datasets/y10', 'rb') as f:
    #     y10 = pk.load(f)

    # uint-3
    # with open('./datasets/y11', 'rb') as f:
    #     y11 = pk.load(f)
    # with open('./datasets/y12', 'rb') as f:
    #     y12 = pk.load(f)
    with open('./datasets/y13', 'rb') as f:
        y13 = pk.load(f)
    with open('./datasets/y14', 'rb') as f:
        y14 = pk.load(f)
    with open('./datasets/y15', 'rb') as f:
        y15 = pk.load(f)

    # uint-4
    with open('./datasets/y16', 'rb') as f:
        y16 = pk.load(f)
    with open('./datasets/y17', 'rb') as f:
        y17 = pk.load(f)
    with open('./datasets/y18', 'rb') as f:
        y18 = pk.load(f)
    with open('./datasets/y19', 'rb') as f:
        y19 = pk.load(f)
    with open('./datasets/y20', 'rb') as f:
        y20 = pk.load(f)
    with open('./datasets/y21', 'rb') as f:
        y21 = pk.load(f)

    xs = []
    m = input.shape[0]
    nums_batches = m // batch_size
    print(input.shape)
    print(nums_batches)
    for k in range(0, nums_batches):
        train_xs = input[batch_size * k: batch_size * (k + 1)]

        # ys6 = y6[batch_size * k: batch_size * (k + 1)]
        # ys7 = y7[batch_size * k: batch_size * (k + 1)]
        # ys8 = y8[batch_size * k: batch_size * (k + 1)]
        # ys9 = y9[batch_size * k: batch_size * (k + 1)]
        # ys10 = y10[batch_size * k: batch_size * (k + 1)]

        # ys11 = y11[batch_size * k: batch_size * (k + 1)]
        # ys12 = y12[batch_size * k: batch_size * (k + 1)]
        ys13 = y13[batch_size * k: batch_size * (k + 1)]
        ys14 = y14[batch_size * k: batch_size * (k + 1)]
        ys15 = y15[batch_size * k: batch_size * (k + 1)]

        ys16 = y16[batch_size * k: batch_size * (k + 1)]
        ys17 = y17[batch_size * k: batch_size * (k + 1)]
        ys18 = y18[batch_size * k: batch_size * (k + 1)]
        ys19 = y19[batch_size * k: batch_size * (k + 1)]
        ys20 = y20[batch_size * k: batch_size * (k + 1)]

        ys21 = y21[batch_size * k: batch_size * (k + 1)]

        xs.append((train_xs, ys13, ys14, ys15, ys16, ys17, ys18, ys19, ys20, ys21))

    return xs, nums_batches

def model_parameters_dataset3(batch_size, index):

    with open('./xs_layers_12345_{}'.format(index), 'rb') as f:
        data = pk.load(f)

    train_data = data
    xs = []
    m = train_data.shape[0]
    nums_batches = m // batch_size
    for k in range(0, nums_batches):
        train_xs = train_data[batch_size * k: batch_size * (k + 1)]
        xs.append(train_xs)

#     print("Complete dataset loading! The shape of dataset is {}. Start the \"Predict Network\" training!"
#           .format(data.shape))

    return xs, nums_batches


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch &lt; mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def load_dataset():

    np.random.seed(1)
    cifar10 = Cifar10(path=r"cifar-10-batches-py", one_hot=True)
    cifar10._load_data()

    # 训练集
    train_xs = cifar10.images / 255.0
    train_labels = cifar10.labels
    print(train_xs.shape)

    # 准备测试集
    test_xs = cifar10.test.images / 255.0
    test_labels = cifar10.test.labels
    print(test_xs.shape)

    return test_xs, test_labels


def create_placeholder():

    # weights samples
    x = tf.placeholder(tf.float32, [None, 8, 8, 5361], name='input_x')

    # images
    images_input = tf.placeholder(tf.float32, [None, 32, 32, 3], name='images_input')
    images_label_a = tf.placeholder(tf.float32, [None, 10], name='y_a')
    images_label_b = tf.placeholder(tf.float32, [None, 10], name='y_b')
    # mixup hyper-parameters
    lam_placeholder = tf.placeholder(tf.float32, name='lam')

    # weight hyper-parameters
    beta = tf.placeholder(tf.float32, name='beta')
    random_seed = tf.placeholder(tf.int32, name='random_seed')

    return x, images_input, images_label_a, images_label_b, lam_placeholder, random_seed


def forward_propagation(input_tensor, input_images, seed):

    is_train = tf.placeholder_with_default(False, (), 'is_train')
    # 1-prediction network
    # Layer-1: Conv-(3, 3, 1188, 6)
    with tf.variable_scope('layer1-Conv1'):
        w1 = tf.get_variable('weight', [3, 3, 5361, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('biases', [8], initializer=tf.constant_initializer(0.1))
        z1 = tf.nn.bias_add(tf.nn.conv2d(input_tensor, w1, strides=[1, 1, 1, 1], padding='SAME'), b1)
        # z1 = tf.layers.batch_normalization(z1, training=is_train)
        a1 = tf.nn.relu(z1)
        a1 = SE_block(a1, ratio=4)

    # Pool layer
    with tf.variable_scope('pool1-Layer'):
        pool1 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer-2: Conv-(3, 3, 6, 8)
    with tf.variable_scope('layer2-Conv2'):
        w2 = tf.get_variable('weight', [3, 3, 8, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('biases', [8], initializer=tf.constant_initializer(0.1))
        z2 = tf.nn.bias_add(tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME'), b2)
        # z2 = tf.layers.batch_normalization(z2, training=is_train)
        a2 = tf.nn.relu(z2)
        a2 = SE_block(a2, ratio=4)

    # Pool layer
    with tf.variable_scope('pool2-Layer'):
        pool2 = tf.nn.max_pool(a2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer-3: Conv-(3, 3, 8, 16)
    with tf.variable_scope('layer3-Conv3'):
        w3 = tf.get_variable('weight', [3, 3, 8, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b3 = tf.get_variable('biases', [8], initializer=tf.constant_initializer(0.1))
        z3 = tf.nn.bias_add(tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding='SAME'), b3)
        # z3 = tf.layers.batch_normalization(z3, training=is_train)
        a3 = tf.nn.relu(z3)
        a3 = SE_block(a3, ratio=4)

    # Pool layer
    with tf.variable_scope('pool3-Layer'):
        pool3 = tf.nn.max_pool(a3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    flatten = tf.contrib.layers.flatten(pool3)

    # Decoder-1: decode vgg11 convolution layer (L3-L8)
    with tf.variable_scope('Decoder-1'):
        w4 = tf.get_variable('decoder-weight-1', [flatten.shape[1], 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b4 = tf.get_variable('decoder-biases-1', [64], initializer=tf.constant_initializer(0.1))
        z4 = tf.matmul(flatten, w4) + b4
        # z4 = tf.layers.batch_normalization(z4, training=is_train2)
        a4 = tf.nn.relu(z4)
        a4 = tf.reshape(a4, [-1, 8, 8, 1])

        # unit-2-3
        w24 = tf.get_variable('conv24-weight', [1, 1, 1, 260], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z24 = tf.nn.conv2d(a4, w24, strides=[1, 1, 1, 1], padding='VALID')
        y24_hat = tf.reshape(z24, [-1, 16640])

        # unit-3
        # shortcut3
        w25 = tf.get_variable('conv25-weight', [1, 1, 1, 2056], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z25 = tf.nn.conv2d(a4, w25, strides=[1, 1, 1, 1], padding='VALID')
        y25_hat = tf.reshape(z25, [-1, 131584])

        # unit-3-0
        w26 = tf.get_variable('conv26-weight', [1, 1, 1, 514], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z26 = tf.nn.conv2d(a4, w26, strides=[1, 1, 1, 1], padding='VALID')
        y26_hat = tf.reshape(z26, [-1, 32896])

        w27 = tf.get_variable('conv27-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z27 = tf.nn.conv2d(a4, w27, strides=[1, 1, 1, 1], padding='VALID')
        y27_hat = tf.reshape(z27, [-1, 147584])

        w28 = tf.get_variable('conv28-weight', [1, 1, 1, 1032], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z28 = tf.nn.conv2d(a4, w28, strides=[1, 1, 1, 1], padding='VALID')
        y28_hat = tf.reshape(z28, [-1, 66048])

        # unit-3-1
        w29 = tf.get_variable('conv29-weight', [1, 1, 1, 1026], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z29 = tf.nn.conv2d(a4, w29, strides=[1, 1, 1, 1], padding='VALID')
        y29_hat = tf.reshape(z29, [-1, 65664])

        w30 = tf.get_variable('conv30-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z30 = tf.nn.conv2d(a4, w30, strides=[1, 1, 1, 1], padding='VALID')
        y30_hat = tf.reshape(z30, [-1, 147584])

        w31 = tf.get_variable('conv31-weight', [1, 1, 1, 1032], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z31 = tf.nn.conv2d(a4, w31, strides=[1, 1, 1, 1], padding='VALID')
        y31_hat = tf.reshape(z31, [-1, 66048])

        # unit-3-2
        w32 = tf.get_variable('conv32-weight', [1, 1, 1, 1026], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z32 = tf.nn.conv2d(a4, w32, strides=[1, 1, 1, 1], padding='VALID')
        y32_hat = tf.reshape(z32, [-1, 65664])

        w33 = tf.get_variable('conv33-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z33 = tf.nn.conv2d(a4, w33, strides=[1, 1, 1, 1], padding='VALID')
        y33_hat = tf.reshape(z33, [-1, 147584])

        w34 = tf.get_variable('conv34-weight', [1, 1, 1, 1032], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z34 = tf.nn.conv2d(a4, w34, strides=[1, 1, 1, 1], padding='VALID')
        y34_hat = tf.reshape(z34, [-1, 66048])

        # unit-3-3
        w35 = tf.get_variable('conv35-weight', [1, 1, 1, 1026], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z35 = tf.nn.conv2d(a4, w35, strides=[1, 1, 1, 1], padding='VALID')
        y35_hat = tf.reshape(z35, [-1, 65664])

        w36 = tf.get_variable('conv36-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z36 = tf.nn.conv2d(a4, w36, strides=[1, 1, 1, 1], padding='VALID')
        y36_hat = tf.reshape(z36, [-1, 147584])

        w37 = tf.get_variable('conv37-weight', [1, 1, 1, 1032], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z37 = tf.nn.conv2d(a4, w37, strides=[1, 1, 1, 1], padding='VALID')
        y37_hat = tf.reshape(z37, [-1, 66048])

        # unit-3-4
        w38 = tf.get_variable('conv38-weight', [1, 1, 1, 1026], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z38 = tf.nn.conv2d(a4, w38, strides=[1, 1, 1, 1], padding='VALID')
        y38_hat = tf.reshape(z38, [-1, 65664])

        w39 = tf.get_variable('conv39-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z39 = tf.nn.conv2d(a4, w39, strides=[1, 1, 1, 1], padding='VALID')
        y39_hat = tf.reshape(z39, [-1, 147584])

        w40 = tf.get_variable('conv40-weight', [1, 1, 1, 1032], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z40 = tf.nn.conv2d(a4, w40, strides=[1, 1, 1, 1], padding='VALID')
        y40_hat = tf.reshape(z40, [-1, 66048])

        # unit-3-5
        w41 = tf.get_variable('conv41-weight', [1, 1, 1, 1026], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z41 = tf.nn.conv2d(a4, w41, strides=[1, 1, 1, 1], padding='VALID')
        y41_hat = tf.reshape(z41, [-1, 65664])

        w42 = tf.get_variable('conv42-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z42 = tf.nn.conv2d(a4, w42, strides=[1, 1, 1, 1], padding='VALID')
        y42_hat = tf.reshape(z42, [-1, 147584])

        w43 = tf.get_variable('conv43-weight', [1, 1, 1, 1032], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z43 = tf.nn.conv2d(a4, w43, strides=[1, 1, 1, 1], padding='VALID')
        y43_hat = tf.reshape(z43, [-1, 66048])

        # unit-4
        # shortcut4
        w44 = tf.get_variable('conv44-weight', [1, 1, 1, 8208], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z44 = tf.nn.conv2d(a4, w44, strides=[1, 1, 1, 1], padding='VALID')
        y44_hat = tf.reshape(z44, [-1, 525312])

        # unit-4-0
        w45 = tf.get_variable('conv45-weight', [1, 1, 1, 2052], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z45 = tf.nn.conv2d(a4, w45, strides=[1, 1, 1, 1], padding='VALID')
        y45_hat = tf.reshape(z45, [-1, 131328])

        w46 = tf.get_variable('conv46-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z46 = tf.nn.conv2d(a4, w46, strides=[1, 1, 1, 1], padding='VALID')
        y46_hat = tf.reshape(z46, [-1, 590080])

        w47 = tf.get_variable('conv47-weight', [1, 1, 1, 4112], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z47 = tf.nn.conv2d(a4, w47, strides=[1, 1, 1, 1], padding='VALID')
        y47_hat = tf.reshape(z47, [-1, 263168])

        # unit-4-1
        w48 = tf.get_variable('conv48-weight', [1, 1, 1, 4100], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z48 = tf.nn.conv2d(a4, w48, strides=[1, 1, 1, 1], padding='VALID')
        y48_hat = tf.reshape(z48, [-1, 262400])

        w49 = tf.get_variable('conv49-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z49 = tf.nn.conv2d(a4, w49, strides=[1, 1, 1, 1], padding='VALID')
        y49_hat = tf.reshape(z49, [-1, 590080])

        w50 = tf.get_variable('conv50-weight', [1, 1, 1, 4112], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z50 = tf.nn.conv2d(a4, w50, strides=[1, 1, 1, 1], padding='VALID')
        y50_hat = tf.reshape(z50, [-1, 263168])

        # unit-4-2
        w51 = tf.get_variable('conv51-weight', [1, 1, 1, 4100], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z51 = tf.nn.conv2d(a4, w51, strides=[1, 1, 1, 1], padding='VALID')
        y51_hat = tf.reshape(z51, [-1, 262400])

        w52 = tf.get_variable('conv52-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z52 = tf.nn.conv2d(a4, w52, strides=[1, 1, 1, 1], padding='VALID')
        y52_hat = tf.reshape(z52, [-1, 590080])

        w53 = tf.get_variable('conv53-weight', [1, 1, 1, 4112], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z53 = tf.nn.conv2d(a4, w53, strides=[1, 1, 1, 1], padding='VALID')
        y53_hat = tf.reshape(z53, [-1, 263168])


    # Decoder-2: decode fcn layer
    with tf.variable_scope('Decoder-2'):
        w54 = tf.get_variable('fcn1-weight', [flatten.shape[1], 10250], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z54 = tf.matmul(flatten, w54)
        y54_hat = tf.reshape(z54, [-1, 10250])

    # 输入切片
    # input_tensor = tf.reduce_mean(input_tensor, axis=0, keep_dims=True)
    input_tensor = tf.reshape(input_tensor, [-1, 343104])

    # init_conv
    eval_w1 = input_tensor[seed, 0: 864]
    eval_w1 = tf.reshape(eval_w1, [3, 3, 3, 32])
    eval_b1 = input_tensor[seed, 864: 896]

    # unit-1
    # shortcut1
    eval_w2 = input_tensor[seed, 896: 4992]
    eval_w2 = tf.reshape(eval_w2, [1, 1, 32, 128])
    eval_b2 = input_tensor[seed, 4992: 5120]

    # unit-1-0
    eval_w3 = input_tensor[seed, 5120: 6144]
    eval_w3 = tf.reshape(eval_w3, [1, 1, 32, 32])
    eval_b3 = input_tensor[seed, 6144: 6176]

    eval_w4 = input_tensor[seed, 6176: 15392]
    eval_w4 = tf.reshape(eval_w4, [3, 3, 32, 32])
    eval_b4 = input_tensor[seed, 15392: 15424]

    eval_w5 = input_tensor[seed, 15424: 19520]
    eval_w5 = tf.reshape(eval_w5, [1, 1, 32, 128])
    eval_b5 = input_tensor[seed, 19520: 19648]

    # unit-1-1
    eval_w6 = input_tensor[seed, 19648: 23744]
    eval_w6 = tf.reshape(eval_w6, [1, 1, 128, 32])
    eval_b6 = input_tensor[seed, 23744: 23776]

    eval_w7 = input_tensor[seed, 23776: 32992]
    eval_w7 = tf.reshape(eval_w7, [3, 3, 32, 32])
    eval_b7 = input_tensor[seed, 32992: 33024]

    eval_w8 = input_tensor[seed, 33024: 37120]
    eval_w8 = tf.reshape(eval_w8, [1, 1, 32, 128])
    eval_b8 = input_tensor[seed, 37120: 37248]

    # unit-1-2
    eval_w9 = input_tensor[seed, 37248: 41344]
    eval_w9 = tf.reshape(eval_w9, [1, 1, 128, 32])
    eval_b9 = input_tensor[seed, 41344: 41376]

    eval_w10 = input_tensor[seed, 41376: 50592]
    eval_w10 = tf.reshape(eval_w10, [3, 3, 32, 32])
    eval_b10 = input_tensor[seed, 50592: 50624]

    eval_w11 = input_tensor[seed, 50624: 54720]
    eval_w11 = tf.reshape(eval_w11, [1, 1, 32, 128])
    eval_b11 = input_tensor[seed, 54720: 54848]

    # unit-2
    # shortcut2
    eval_w12 = input_tensor[seed, 54848: 87616]
    eval_w12 = tf.reshape(eval_w12, [1, 1, 128, 256])
    eval_b12 = input_tensor[seed, 87616: 87872]

    # unit-2-0
    eval_w13 = input_tensor[seed, 87872: 96064]
    eval_w13 = tf.reshape(eval_w13, [1, 1, 128, 64])
    eval_b13 = input_tensor[seed, 96064: 96128]

    eval_w14 = input_tensor[seed, 96128: 132992]
    eval_w14 = tf.reshape(eval_w14, [3, 3, 64, 64])
    eval_b14 = input_tensor[seed, 132992: 133056]

    eval_w15 = input_tensor[seed, 133056: 149440]
    eval_w15 = tf.reshape(eval_w15, [1, 1, 64, 256])
    eval_b15 = input_tensor[seed, 149440: 149696]

    # unit-2-1
    eval_w16 = input_tensor[seed, 149696: 166080]
    eval_w16 = tf.reshape(eval_w16, [1, 1, 256, 64])
    eval_b16 = input_tensor[seed, 166080: 166144]

    eval_w17 = input_tensor[seed, 166144: 203008]
    eval_w17 = tf.reshape(eval_w17, [3, 3, 64, 64])
    eval_b17 = input_tensor[seed, 203008: 203072]

    eval_w18 = input_tensor[seed, 203072: 219456]
    eval_w18 = tf.reshape(eval_w18, [1, 1, 64, 256])
    eval_b18 = input_tensor[seed, 219456: 219712]

    # unit-2-2
    eval_w19 = input_tensor[seed, 219712: 236096]
    eval_w19 = tf.reshape(eval_w19, [1, 1, 256, 64])
    eval_b19 = input_tensor[seed, 236096: 236160]

    eval_w20 = input_tensor[seed, 236160: 273024]
    eval_w20 = tf.reshape(eval_w20, [3, 3, 64, 64])
    eval_b20 = input_tensor[seed, 273024: 273088]

    eval_w21 = input_tensor[seed, 273088: 289472]
    eval_w21 = tf.reshape(eval_w21, [1, 1, 64, 256])
    eval_b21 = input_tensor[seed, 289472: 289728]

    # unit-2-3
    eval_w22 = input_tensor[seed, 289728: 306112]
    eval_w22 = tf.reshape(eval_w22, [1, 1, 256, 64])
    eval_b22 = input_tensor[seed, 306112: 306176]

    eval_w23 = input_tensor[seed, 306176: 343040]
    eval_w23 = tf.reshape(eval_w23, [3, 3, 64, 64])
    eval_b23 = input_tensor[seed, 343040: 343104]

    # uint-2-3
    decoder_conv_w24 = y24_hat[seed, 0: 16384]
    decoder_conv_w24 = tf.reshape(decoder_conv_w24, [1, 1, 64, 256])
    decoder_conv_b24 = y24_hat[seed, 16384: 16640]

    # unit-3
    # shortcut3
    decoder_conv_w25 = y25_hat[seed, 0: 131072]
    decoder_conv_w25 = tf.reshape(decoder_conv_w25, [1, 1, 256, 512])
    decoder_conv_b25 = y25_hat[seed, 131072: 131584]

    # unit-3-0
    decoder_conv_w26 = y26_hat[seed, 0: 32768]
    decoder_conv_w26 = tf.reshape(decoder_conv_w26, [1, 1, 256, 128])
    decoder_conv_b26 = y26_hat[seed, 32768: 32896]

    decoder_conv_w27 = y27_hat[seed, 0: 147456]
    decoder_conv_w27 = tf.reshape(decoder_conv_w27, [3, 3, 128, 128])
    decoder_conv_b27 = y27_hat[seed, 147456: 147584]

    decoder_conv_w28 = y28_hat[seed, 0: 65536]
    decoder_conv_w28 = tf.reshape(decoder_conv_w28, [1, 1, 128, 512])
    decoder_conv_b28 = y28_hat[seed, 65536: 66048]

    # unit-3-1
    decoder_conv_w29 = y29_hat[seed, 0: 65536]
    decoder_conv_w29 = tf.reshape(decoder_conv_w29, [1, 1, 512, 128])
    decoder_conv_b29 = y29_hat[seed, 65536: 65664]

    decoder_conv_w30 = y30_hat[seed, 0: 147456]
    decoder_conv_w30 = tf.reshape(decoder_conv_w30, [3, 3, 128, 128])
    decoder_conv_b30 = y30_hat[seed, 147456: 147584]

    decoder_conv_w31 = y31_hat[seed, 0: 65536]
    decoder_conv_w31 = tf.reshape(decoder_conv_w31, [1, 1, 128, 512])
    decoder_conv_b31 = y31_hat[seed, 65536: 66048]

    # unit-3-2
    decoder_conv_w32 = y32_hat[seed, 0: 65536]
    decoder_conv_w32 = tf.reshape(decoder_conv_w32, [1, 1, 512, 128])
    decoder_conv_b32 = y32_hat[seed, 65536:  65664]

    decoder_conv_w33 = y33_hat[seed, 0: 147456]
    decoder_conv_w33 = tf.reshape(decoder_conv_w33, [3, 3, 128, 128])
    decoder_conv_b33 = y33_hat[seed, 147456: 147584]

    decoder_conv_w34 = y34_hat[seed, 0: 65536]
    decoder_conv_w34 = tf.reshape(decoder_conv_w34, [1, 1, 128, 512])
    decoder_conv_b34 = y34_hat[seed, 65536: 66048]

    # unit-3-3
    decoder_conv_w35 = y35_hat[seed, 0: 65536]
    decoder_conv_w35 = tf.reshape(decoder_conv_w35, [1, 1, 512, 128])
    decoder_conv_b35 = y35_hat[seed, 65536:  65664]

    decoder_conv_w36 = y36_hat[seed, 0: 147456]
    decoder_conv_w36 = tf.reshape(decoder_conv_w36, [3, 3, 128, 128])
    decoder_conv_b36 = y36_hat[seed, 147456: 147584]

    decoder_conv_w37 = y37_hat[seed, 0: 65536]
    decoder_conv_w37 = tf.reshape(decoder_conv_w37, [1, 1, 128, 512])
    decoder_conv_b37 = y37_hat[seed, 65536: 66048]

    # unit-3-4
    decoder_conv_w38 = y38_hat[seed, 0: 65536]
    decoder_conv_w38 = tf.reshape(decoder_conv_w38, [1, 1, 512, 128])
    decoder_conv_b38 = y38_hat[seed, 65536:  65664]

    decoder_conv_w39 = y39_hat[seed, 0: 147456]
    decoder_conv_w39 = tf.reshape(decoder_conv_w39, [3, 3, 128, 128])
    decoder_conv_b39 = y39_hat[seed, 147456: 147584]

    decoder_conv_w40 = y40_hat[seed, 0: 65536]
    decoder_conv_w40 = tf.reshape(decoder_conv_w40, [1, 1, 128, 512])
    decoder_conv_b40 = y40_hat[seed, 65536: 66048]

    # unit-3-5
    decoder_conv_w41 = y41_hat[seed, 0: 65536]
    decoder_conv_w41 = tf.reshape(decoder_conv_w41, [1, 1, 512, 128])
    decoder_conv_b41 = y41_hat[seed, 65536:  65664]

    decoder_conv_w42 = y42_hat[seed, 0: 147456]
    decoder_conv_w42 = tf.reshape(decoder_conv_w42, [3, 3, 128, 128])
    decoder_conv_b42 = y42_hat[seed, 147456: 147584]

    decoder_conv_w43 = y43_hat[seed, 0: 65536]
    decoder_conv_w43 = tf.reshape(decoder_conv_w43, [1, 1, 128, 512])
    decoder_conv_b43 = y43_hat[seed, 65536: 66048]

    # unit-4
    # shortcut4
    decoder_conv_w44 = y44_hat[seed, 0: 524288]
    decoder_conv_w44 = tf.reshape(decoder_conv_w44, [1, 1, 512, 1024])
    decoder_conv_b44 = y44_hat[seed, 524288: 525312]

    # unit-4-0
    decoder_conv_w45 = y45_hat[seed, 0: 131072]
    decoder_conv_w45 = tf.reshape(decoder_conv_w45, [1, 1, 512, 256])
    decoder_conv_b45 = y45_hat[seed, 131072:  131328]

    decoder_conv_w46 = y46_hat[seed, 0: 589824]
    decoder_conv_w46 = tf.reshape(decoder_conv_w46, [3, 3, 256, 256])
    decoder_conv_b46 = y46_hat[seed, 589824: 590080]

    decoder_conv_w47 = y47_hat[seed, 0: 262144]
    decoder_conv_w47 = tf.reshape(decoder_conv_w47, [1, 1, 256, 1024])
    decoder_conv_b47 = y47_hat[seed, 262144: 263168]

    # unit-4-1
    decoder_conv_w48 = y48_hat[seed, 0: 262144]
    decoder_conv_w48 = tf.reshape(decoder_conv_w48, [1, 1, 1024, 256])
    decoder_conv_b48 = y48_hat[seed, 262144:  262400]

    decoder_conv_w49 = y49_hat[seed, 0: 589824]
    decoder_conv_w49 = tf.reshape(decoder_conv_w49, [3, 3, 256, 256])
    decoder_conv_b49 = y49_hat[seed, 589824: 590080]

    decoder_conv_w50 = y50_hat[seed, 0: 262144]
    decoder_conv_w50 = tf.reshape(decoder_conv_w50, [1, 1, 256, 1024])
    decoder_conv_b50 = y50_hat[seed, 262144: 263168]

    # unit-4-2
    decoder_conv_w51 = y51_hat[seed, 0: 262144]
    decoder_conv_w51 = tf.reshape(decoder_conv_w51, [1, 1, 1024, 256])
    decoder_conv_b51 = y51_hat[seed, 262144:  262400]

    decoder_conv_w52 = y52_hat[seed, 0: 589824]
    decoder_conv_w52= tf.reshape(decoder_conv_w52, [3, 3, 256, 256])
    decoder_conv_b52 = y52_hat[seed, 589824: 590080]

    decoder_conv_w53 = y53_hat[seed, 0: 262144]
    decoder_conv_w53 = tf.reshape(decoder_conv_w53, [1, 1, 256, 1024])
    decoder_conv_b53 = y53_hat[seed, 262144: 263168]

    decoder_fcn_w54 = y54_hat[seed, 0: 10240]
    decoder_fcn_w54 = tf.reshape(decoder_fcn_w54, [1024, 10])
    decoder_fcn_b54 = y54_hat[seed, 10240: 10250]

    # 2-evaluate network
    # evaluate CIFAR10 images with the predicted weight parameters

    # init-conv
    eval_conv1 = tf.nn.conv2d(input_images, eval_w1, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv1 = tf.nn.bias_add(eval_conv1, eval_b1)
    eval_relu1 = tf.nn.relu(eval_conv1)

    eval_pool1 = tf.nn.max_pool(eval_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # unit-1
    # shortcut1
    eval_conv2 = tf.nn.conv2d(eval_pool1, eval_w2, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv2 = tf.nn.bias_add(eval_conv2, eval_b2)

    # unit-1-0
    eval_conv3 = tf.nn.conv2d(eval_pool1, eval_w3, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv3 = tf.nn.bias_add(eval_conv3, eval_b3)
    eval_relu3 = tf.nn.relu(eval_conv3)

    eval_conv4 = tf.nn.conv2d(eval_relu3, eval_w4, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv4 = tf.nn.bias_add(eval_conv4, eval_b4)
    eval_relu4 = tf.nn.relu(eval_conv4)

    eval_conv5 = tf.nn.conv2d(eval_relu4, eval_w5, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv5 = tf.nn.bias_add(eval_conv5, eval_b5)
    eval_relu5 = tf.nn.relu(eval_conv5+eval_conv2)

    # unit-1-1
    eval_conv6 = tf.nn.conv2d(eval_relu5, eval_w6, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv6 = tf.nn.bias_add(eval_conv6, eval_b6)
    eval_relu6 = tf.nn.relu(eval_conv6)

    eval_conv7 = tf.nn.conv2d(eval_relu6, eval_w7, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv7 = tf.nn.bias_add(eval_conv7, eval_b7)
    eval_relu7 = tf.nn.relu(eval_conv7)

    eval_conv8 = tf.nn.conv2d(eval_relu7, eval_w8, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv8 = tf.nn.bias_add(eval_conv8, eval_b8)
    eval_relu8 = tf.nn.relu(eval_conv8+eval_relu5)

    # unit-1-2
    eval_conv9 = tf.nn.conv2d(eval_relu8, eval_w9, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv9 = tf.nn.bias_add(eval_conv9, eval_b9)
    eval_relu9 = tf.nn.relu(eval_conv9)

    eval_conv10 = tf.nn.conv2d(eval_relu9, eval_w10, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv10 = tf.nn.bias_add(eval_conv10, eval_b10)
    eval_relu10 = tf.nn.relu(eval_conv10)

    eval_conv11 = tf.nn.conv2d(eval_relu10, eval_w11, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv11 = tf.nn.bias_add(eval_conv11, eval_b11)
    eval_relu11 = tf.nn.relu(eval_conv11+eval_relu8)

    # unit-2
    # shortcut2
    eval_conv12 = tf.nn.conv2d(eval_relu11, eval_w12, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv12 = tf.nn.bias_add(eval_conv12, eval_b12)

    # unit-2-0
    eval_conv13 = tf.nn.conv2d(eval_relu11, eval_w13, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv13 = tf.nn.bias_add(eval_conv13, eval_b13)
    eval_relu13 = tf.nn.relu(eval_conv13)

    eval_conv14 = tf.nn.conv2d(eval_relu13, eval_w14, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv14 = tf.nn.bias_add(eval_conv14, eval_b14)
    eval_relu14 = tf.nn.relu(eval_conv14)

    eval_conv15 = tf.nn.conv2d(eval_relu14, eval_w15, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv15 = tf.nn.bias_add(eval_conv15, eval_b15)
    eval_relu15 = tf.nn.relu(eval_conv15+eval_conv12)

    # unit-2-1
    eval_conv16 = tf.nn.conv2d(eval_relu15, eval_w16, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv16 = tf.nn.bias_add(eval_conv16, eval_b16)
    eval_relu16 = tf.nn.relu(eval_conv16)

    eval_conv17 = tf.nn.conv2d(eval_relu16, eval_w17, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv17 = tf.nn.bias_add(eval_conv17, eval_b17)
    eval_relu17 = tf.nn.relu(eval_conv17)

    eval_conv18 = tf.nn.conv2d(eval_relu17, eval_w18, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv18 = tf.nn.bias_add(eval_conv18, eval_b18)
    eval_relu18 = tf.nn.relu(eval_conv18+eval_relu15)

    # unit-2-2
    eval_conv19 = tf.nn.conv2d(eval_relu18, eval_w19, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv19 = tf.nn.bias_add(eval_conv19, eval_b19)
    eval_relu19 = tf.nn.relu(eval_conv19)

    eval_conv20 = tf.nn.conv2d(eval_relu19, eval_w20, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv20 = tf.nn.bias_add(eval_conv20, eval_b20)
    eval_relu20 = tf.nn.relu(eval_conv20)

    eval_conv21 = tf.nn.conv2d(eval_relu20, eval_w21, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv21 = tf.nn.bias_add(eval_conv21, eval_b21)
    eval_relu21 = tf.nn.relu(eval_conv21 + eval_relu18)

    # unit-2-2
    eval_conv22 = tf.nn.conv2d(eval_relu21, eval_w22, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv22 = tf.nn.bias_add(eval_conv22, eval_b22)
    eval_relu22= tf.nn.relu(eval_conv22)

    eval_conv23 = tf.nn.conv2d(eval_relu22, eval_w23, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv23 = tf.nn.bias_add(eval_conv23, eval_b23)
    eval_relu23 = tf.nn.relu(eval_conv23)

    eval_conv24 = tf.nn.conv2d(eval_relu23, decoder_conv_w24, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv24 = tf.nn.bias_add(eval_conv24, decoder_conv_b24)
    eval_relu24 = tf.nn.relu(eval_conv24 + eval_relu21)

    # uint-3
    # shortcut
    eval_conv25 = tf.nn.conv2d(eval_relu24, decoder_conv_w25, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv25 = tf.nn.bias_add(eval_conv25, decoder_conv_b25)

    # unit-3-0
    eval_conv26 = tf.nn.conv2d(eval_relu24, decoder_conv_w26, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv26 = tf.nn.bias_add(eval_conv26, decoder_conv_b26)
    eval_relu26 = tf.nn.relu(eval_conv26)

    eval_conv27 = tf.nn.conv2d(eval_relu26, decoder_conv_w27, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv27 = tf.nn.bias_add(eval_conv27, decoder_conv_b27)
    eval_relu27 = tf.nn.relu(eval_conv27)

    eval_conv28 = tf.nn.conv2d(eval_relu27, decoder_conv_w28, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv28 = tf.nn.bias_add(eval_conv28, decoder_conv_b28)
    eval_relu28 = tf.nn.relu(eval_conv28 + eval_conv25)

    # unit-3-1
    eval_conv29 = tf.nn.conv2d(eval_relu28, decoder_conv_w29, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv29 = tf.nn.bias_add(eval_conv29, decoder_conv_b29)
    eval_relu29= tf.nn.relu(eval_conv29)

    eval_conv30 = tf.nn.conv2d(eval_relu29, decoder_conv_w30, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv30 = tf.nn.bias_add(eval_conv30, decoder_conv_b30)
    eval_relu30 = tf.nn.relu(eval_conv30)

    eval_conv31 = tf.nn.conv2d(eval_relu30, decoder_conv_w31, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv31 = tf.nn.bias_add(eval_conv31, decoder_conv_b31)
    eval_relu31 = tf.nn.relu(eval_conv31 + eval_relu28)

    # unit-3-2
    eval_conv32 = tf.nn.conv2d(eval_relu31, decoder_conv_w32, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv32 = tf.nn.bias_add(eval_conv32, decoder_conv_b32)
    eval_relu32 = tf.nn.relu(eval_conv32)

    eval_conv33 = tf.nn.conv2d(eval_relu32, decoder_conv_w33, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv33 = tf.nn.bias_add(eval_conv33, decoder_conv_b33)
    eval_relu33 = tf.nn.relu(eval_conv33)

    eval_conv34 = tf.nn.conv2d(eval_relu33, decoder_conv_w34, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv34 = tf.nn.bias_add(eval_conv34, decoder_conv_b34)
    eval_relu34 = tf.nn.relu(eval_conv34 + eval_relu31)

    # unit-3-3
    eval_conv35 = tf.nn.conv2d(eval_relu34, decoder_conv_w35, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv35 = tf.nn.bias_add(eval_conv35, decoder_conv_b35)
    eval_relu35 = tf.nn.relu(eval_conv35)

    eval_conv36 = tf.nn.conv2d(eval_relu35, decoder_conv_w36, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv36 = tf.nn.bias_add(eval_conv36, decoder_conv_b36)
    eval_relu36 = tf.nn.relu(eval_conv36)

    eval_conv37 = tf.nn.conv2d(eval_relu36, decoder_conv_w37, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv37 = tf.nn.bias_add(eval_conv37, decoder_conv_b37)
    eval_relu37 = tf.nn.relu(eval_conv37 + eval_relu34)

    # unit-3-4
    eval_conv38 = tf.nn.conv2d(eval_relu37, decoder_conv_w38, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv38 = tf.nn.bias_add(eval_conv38, decoder_conv_b38)
    eval_relu38 = tf.nn.relu(eval_conv38)

    eval_conv39 = tf.nn.conv2d(eval_relu38, decoder_conv_w39, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv39 = tf.nn.bias_add(eval_conv39, decoder_conv_b39)
    eval_relu39 = tf.nn.relu(eval_conv39)

    eval_conv40 = tf.nn.conv2d(eval_relu39, decoder_conv_w40, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv40 = tf.nn.bias_add(eval_conv40, decoder_conv_b40)
    eval_relu40 = tf.nn.relu(eval_conv40+ eval_relu37)

    # unit-3-5
    eval_conv41 = tf.nn.conv2d(eval_relu40, decoder_conv_w41, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv41 = tf.nn.bias_add(eval_conv41, decoder_conv_b41)
    eval_relu41 = tf.nn.relu(eval_conv41)

    eval_conv42 = tf.nn.conv2d(eval_relu41, decoder_conv_w42, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv42 = tf.nn.bias_add(eval_conv42, decoder_conv_b42)
    eval_relu42 = tf.nn.relu(eval_conv42)

    eval_conv43 = tf.nn.conv2d(eval_relu42, decoder_conv_w43, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv43 = tf.nn.bias_add(eval_conv43, decoder_conv_b43)
    eval_relu43 = tf.nn.relu(eval_conv43 + eval_relu40)

    # uint-4
    # shortcut
    eval_conv44 = tf.nn.conv2d(eval_relu43, decoder_conv_w44, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv44 = tf.nn.bias_add(eval_conv44, decoder_conv_b44)

    # unit-4-0
    eval_conv45 = tf.nn.conv2d(eval_relu43, decoder_conv_w45, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv45 = tf.nn.bias_add(eval_conv45, decoder_conv_b45)
    eval_relu45 = tf.nn.relu(eval_conv45)

    eval_conv46 = tf.nn.conv2d(eval_relu45, decoder_conv_w46, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv46 = tf.nn.bias_add(eval_conv46, decoder_conv_b46)
    eval_relu46 = tf.nn.relu(eval_conv46)

    eval_conv47 = tf.nn.conv2d(eval_relu46, decoder_conv_w47, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv47 = tf.nn.bias_add(eval_conv47, decoder_conv_b47)
    eval_relu47 = tf.nn.relu(eval_conv47 + eval_conv44)

    # unit-4-1
    eval_conv48 = tf.nn.conv2d(eval_relu47, decoder_conv_w48, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv48 = tf.nn.bias_add(eval_conv48, decoder_conv_b48)
    eval_relu48 = tf.nn.relu(eval_conv48)

    eval_conv49 = tf.nn.conv2d(eval_relu48, decoder_conv_w49, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv49 = tf.nn.bias_add(eval_conv49, decoder_conv_b49)
    eval_relu49 = tf.nn.relu(eval_conv49)

    eval_conv50 = tf.nn.conv2d(eval_relu49, decoder_conv_w50, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv50 = tf.nn.bias_add(eval_conv50, decoder_conv_b50)
    eval_relu50 = tf.nn.relu(eval_conv50 + eval_relu47)

    # unit-4-2
    eval_conv51 = tf.nn.conv2d(eval_relu50, decoder_conv_w51, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv51 = tf.nn.bias_add(eval_conv51, decoder_conv_b51)
    eval_relu51 = tf.nn.relu(eval_conv51)

    eval_conv52 = tf.nn.conv2d(eval_relu51, decoder_conv_w52, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv52 = tf.nn.bias_add(eval_conv52, decoder_conv_b52)
    eval_relu52 = tf.nn.relu(eval_conv52)

    eval_conv53 = tf.nn.conv2d(eval_relu52, decoder_conv_w53, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv53 = tf.nn.bias_add(eval_conv53, decoder_conv_b53)
    eval_relu53 = tf.nn.relu(eval_conv53 + eval_relu50)

    eval_pool2 = tf.nn.avg_pool(eval_relu53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print(eval_pool2.shape)
    p = tf.contrib.layers.flatten(eval_pool2)

    logits = tf.matmul(p, decoder_fcn_w54) + decoder_fcn_b54


    return logits, is_train


def compute_cost(logits, y_a, y_b, lam, lr, trainable_vars):
    # evaluation loss
    cross_entropy_a = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(y_a, 1))
    cross_entropy_b = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(y_b, 1))
    loss_a = tf.reduce_mean(cross_entropy_a)
    loss_b = tf.reduce_mean(cross_entropy_b)

    # weight loss
    # w_loss = tf.reduce_mean(tf.square(y13 - y13_hat)) + tf.reduce_mean(tf.square(y14 - y14_hat)) + \
    #          tf.reduce_mean(tf.square(y15 - y15_hat)) + tf.reduce_mean(tf.square(y16 - y16_hat)) + \
    #          tf.reduce_mean(tf.square(y17 - y17_hat)) + tf.reduce_mean(tf.square(y18 - y18_hat)) + \
    #          tf.reduce_mean(tf.square(y19 - y19_hat)) + tf.reduce_mean(tf.square(y20 - y20_hat)) + \
    #          tf.reduce_mean(tf.square(y21 - y21_hat))

    # beta hyper-parameters
    # loss1 = w_loss
    loss2 = loss_a * lam + loss_b * (1 - lam)
    # 1、总是取最小损失
    # loss = tf.math.maximum(loss1, loss2)
    # 2、加权的方式
    # loss = loss1*beta + loss2*(1 - beta)
    # loss = loss_a*lam + loss_b*(1 - lam) + beta*w_loss
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss2, var_list=trainable_vars)

    return logits, loss2, optimizer


def compute_accuracy(logits, y):

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def batch_normalization(input, beta, gama, eps=1e-5):

    input_shape = input.get_shape()
    axis = list(range(len(input_shape) - 1))
    mean, var = tf.nn.moments(input, axis)

    return tf.nn.batch_normalization(input, mean, var, beta, gama, eps)

def main(epoch_nums):

    # input_paras, nums_xs_batch = model_parameters_dataset(batch_size=250)

    predict_input, eval_input, eval_ya, eval_yb, lam_tensor, random_seed = create_placeholder()
    # 加入正则化
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZERATION)
    logits, istrain = forward_propagation(predict_input, eval_input, random_seed)
    variables = tf.trainable_variables()
    # trainable_vars = variables[0: 6] + variables[20:]
    # print('trainable_vars:')
    # print(trainable_vars)
    # untrained_vars = variables[6:20]
    # print('untrained_vars:')
    # print(untrained_vars)

    logits, loss, optimizer = compute_cost(logits, eval_ya, eval_yb, lam_tensor, 0.001, variables)
    accuracy = compute_accuracy(logits, eval_ya)

    init = tf.global_variables_initializer()
    # saver1 = tf.train.Saver(untrained_vars)
    saver2 = tf.train.Saver()         # 保存模型

    test_xs, test_ys = load_dataset()
    train_dataset = utils.load_data(path='./cifar-10-classed',)

    costs = []
    test_accs = []

    max_acc = 0
    iterations = 0
    nums_xs_batches = 180

    L = list(np.arange(start=500, stop=5000, step=500))

    with tf.Session() as sess:
        sess.run(init)
        # saver1.restore(sess, './model-resnet/pred_model_for_resnet/model.ckpt')      # 重新加载模型，微调后继续训练
        # saver2.restore(sess, "dataset/save_model1/model.ckpt")
        for epoch in range(1, epoch_nums+1):
            epoch_cost = 0
            sum_of_time = 0
            temp = []

            for l in L:
                input_paras, nums_batches = model_parameters_dataset3(batch_size=25, index=l)

                for xs in input_paras:
                    start = time.time()
                    seed = np.random.randint(0, 25)
                    X = xs.reshape([-1, 8, 8, 5361])
                    iterations_cost = 0
                    iterations += 1
                    # sample 10% images from each class
                    mini_batches, image_batches = utils.sample_image(train_dataset, 10, percent=0.25, seed=iterations, mini_batch_size=125)

                    for mini_batch in mini_batches:
                        (mini_batch_X, mini_batch_Y) = mini_batch
                        # mixup : 混合增强
                        x, mix_x, label_a, label_b, lam = mixup_data(mini_batch_X, mini_batch_Y, alpha=1)
                        _, cost, train_acc = sess.run([optimizer, loss, accuracy],
                                                      feed_dict={predict_input: X, eval_input: mix_x,
                                                                 eval_ya: label_a, eval_yb: label_b,
                                                                 lam_tensor: lam, random_seed: seed
                                                                 })
                        iterations_cost += cost/image_batches
                    end = time.time()
                    sum_of_time += (end - start)
                    test_acc = sess.run(accuracy, feed_dict={predict_input: X, eval_input: test_xs, eval_ya: test_ys,
                                                             random_seed: seed})
                    temp.append(test_acc)

                    if test_acc > max_acc:
                        max_acc = test_acc
                        saver2.save(sess, "./save-model/model.ckpt")    # 保存精度最高时候的模型
                    if iterations % 1 == 0:
                        print("Epoch {}/{}, Iteration {}/{}, Training cost is {}, Test accuracy is {}"
                              .format(epoch, epoch_nums, iterations, epoch_nums * nums_xs_batches, iterations_cost, test_acc))
                    epoch_cost += iterations_cost / nums_xs_batches
            test_accs.append(max(temp))
            costs.append(epoch_cost)
            if epoch % 1 == 0:
                print("After {} Epoch, epoch cost is {}, Max test accuracy is {}, Test accuracy is {}, {} sec/batch"
                      .format(epoch, epoch_cost, max_acc, test_accs[-1], sum_of_time//nums_xs_batches))
            # 保存收敛时的模型
            # saver.save(sess, './save_model/model.ckpt')

    # with open('./s1_layers_12/result/cost', 'wb') as f:
    #     f.write(pk.dumps(costs))
    # with open('./s1_layers_12/result/test_acc', 'wb') as f:
    #     f.write(pk.dumps(test_accs))


if __name__=='__main__':
    main(epoch_nums=30)
