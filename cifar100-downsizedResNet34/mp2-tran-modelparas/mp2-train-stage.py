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

# 正则化系数
REGULARIZERATION = 0.0002


def model_parameters_dataset(batch_size):

    with open('datasets/train_xs', 'rb') as f:
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

    return train_xs, train_labels, test_xs, test_labels


def create_placeholder():

    # weights samples
    x = tf.placeholder(tf.float32, [None, 8, 8, 5498], name='input_x')

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
        w1 = tf.get_variable('weight', [3, 3, 5498, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
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

        # unit-3
        w17 = tf.get_variable('conv11-weight', [1, 1, 1, 1154], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z17 = tf.nn.conv2d(a4, w17, strides=[1, 1, 1, 1], padding='VALID')
        y17_hat = tf.reshape(z17, [-1, 73856])

        w18 = tf.get_variable('conv11-weight', [1, 1, 1, 1154], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z18 = tf.nn.conv2d(a4, w18, strides=[1, 1, 1, 1], padding='VALID')
        y18_hat = tf.reshape(z18, [-1, 73856])

        w19 = tf.get_variable('conv12-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z19 = tf.nn.conv2d(a4, w19, strides=[1, 1, 1, 1], padding='VALID')
        y19_hat = tf.reshape(z19, [-1, 147584])

        w20 = tf.get_variable('conv13-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z20 = tf.nn.conv2d(a4, w20, strides=[1, 1, 1, 1], padding='VALID')
        y20_hat = tf.reshape(z20, [-1, 147584])

        w21 = tf.get_variable('conv14-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z21 = tf.nn.conv2d(a4, w21, strides=[1, 1, 1, 1], padding='VALID')
        y21_hat = tf.reshape(z21, [-1, 147584])

        w22 = tf.get_variable('conv15-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z22 = tf.nn.conv2d(a4, w22, strides=[1, 1, 1, 1], padding='VALID')
        y22_hat = tf.reshape(z22, [-1, 147584])

        w23 = tf.get_variable('conv15-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z23 = tf.nn.conv2d(a4, w23, strides=[1, 1, 1, 1], padding='VALID')
        y23_hat = tf.reshape(z23, [-1, 147584])

        w24 = tf.get_variable('conv15-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z24 = tf.nn.conv2d(a4, w24, strides=[1, 1, 1, 1], padding='VALID')
        y24_hat = tf.reshape(z24, [-1, 147584])

        w25 = tf.get_variable('conv15-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z25 = tf.nn.conv2d(a4, w25, strides=[1, 1, 1, 1], padding='VALID')
        y25_hat = tf.reshape(z25, [-1, 147584])

        w26 = tf.get_variable('conv15-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z26 = tf.nn.conv2d(a4, w26, strides=[1, 1, 1, 1], padding='VALID')
        y26_hat = tf.reshape(z26, [-1, 147584])

        w27 = tf.get_variable('conv15-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z27 = tf.nn.conv2d(a4, w27, strides=[1, 1, 1, 1], padding='VALID')
        y27_hat = tf.reshape(z27, [-1, 147584])

        w28 = tf.get_variable('conv15-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z28 = tf.nn.conv2d(a4, w28, strides=[1, 1, 1, 1], padding='VALID')
        y28_hat = tf.reshape(z28, [-1, 147584])

        w29 = tf.get_variable('conv15-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z29 = tf.nn.conv2d(a4, w29, strides=[1, 1, 1, 1], padding='VALID')
        y29_hat = tf.reshape(z29, [-1, 147584])

        # unit-4
        w30 = tf.get_variable('conv16-weight', [1, 1, 1, 4612], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z30 = tf.nn.conv2d(a4, w30, strides=[1, 1, 1, 1], padding='VALID')
        y30_hat = tf.reshape(z30, [-1, 295168])

        w31 = tf.get_variable('conv17-weight', [1, 1, 1, 4612], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z31 = tf.nn.conv2d(a4, w31, strides=[1, 1, 1, 1], padding='VALID')
        y31_hat = tf.reshape(z31, [-1, 295168])

        w32 = tf.get_variable('conv18-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z32 = tf.nn.conv2d(a4, w32, strides=[1, 1, 1, 1], padding='VALID')
        y32_hat = tf.reshape(z32, [-1, 590080])

        w33 = tf.get_variable('conv19-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z33 = tf.nn.conv2d(a4, w33, strides=[1, 1, 1, 1], padding='VALID')
        y33_hat = tf.reshape(z33, [-1, 590080])

        w34 = tf.get_variable('conv20-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z34 = tf.nn.conv2d(a4, w34, strides=[1, 1, 1, 1], padding='VALID')
        y34_hat = tf.reshape(z34, [-1, 590080])

        w35 = tf.get_variable('conv20-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z35 = tf.nn.conv2d(a4, w35, strides=[1, 1, 1, 1], padding='VALID')
        y35_hat = tf.reshape(z35, [-1, 590080])

        w36 = tf.get_variable('conv20-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z36 = tf.nn.conv2d(a4, w36, strides=[1, 1, 1, 1], padding='VALID')
        y36_hat = tf.reshape(z36, [-1, 590080])

    # Decoder-2: decode fcn layer
    with tf.variable_scope('Decoder-2'):
        w37 = tf.get_variable('fcn1-weight', [flatten.shape[1], 25700], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z37 = tf.matmul(flatten, w37)
        y37_hat = tf.reshape(z37, [-1, 25700])

    # 输入切片
    # input_tensor = tf.reduce_mean(input_tensor, axis=0, keep_dims=True)
    input_tensor = tf.reshape(input_tensor, [-1, 351872])

    # init_conv
    eval_w1 = input_tensor[seed, 0: 864]
    eval_w1 = tf.reshape(eval_w1, [3, 3, 3, 32])
    eval_b1 = input_tensor[seed, 864: 896]

    # unit-1
    eval_w2 = input_tensor[seed, 896: 10112]
    eval_w2 = tf.reshape(eval_w2, [3, 3, 32, 32])
    eval_b2 = input_tensor[seed, 10112: 10144]

    eval_w3 = input_tensor[seed, 10144: 19360]
    eval_w3 = tf.reshape(eval_w3, [3, 3, 32, 32])
    eval_b3 = input_tensor[seed, 19360: 19392]

    eval_w4 = input_tensor[seed, 19392: 28608]
    eval_w4 = tf.reshape(eval_w4, [3, 3, 32, 32])
    eval_b4 = input_tensor[seed, 28608: 28640]

    eval_w5 = input_tensor[seed, 28640: 37856]
    eval_w5 = tf.reshape(eval_w5, [3, 3, 32, 32])
    eval_b5 = input_tensor[seed, 37856: 37888]

    eval_w6 = input_tensor[seed, 37888: 47104]
    eval_w6 = tf.reshape(eval_w6, [3, 3, 32, 32])
    eval_b6 = input_tensor[seed, 47104: 47136]

    eval_w7 = input_tensor[seed, 47136: 56352]
    eval_w7 = tf.reshape(eval_w7, [3, 3, 32, 32])
    eval_b7 = input_tensor[seed, 56352: 56384]

    # unit-2
    eval_w8 = input_tensor[seed, 56384: 74816]
    eval_w8 = tf.reshape(eval_w8, [3, 3, 32, 64])
    eval_b8 = input_tensor[seed, 74816: 74880]

    eval_w9 = input_tensor[seed, 74880: 93312]
    eval_w9 = tf.reshape(eval_w9, [3, 3, 32, 64])
    eval_b9 = input_tensor[seed, 93312: 93376]

    eval_w10 = input_tensor[seed, 93376: 130240]
    eval_w10 = tf.reshape(eval_w10, [3, 3, 64, 64])
    eval_b10 = input_tensor[seed, 130240: 130304]

    eval_w11 = input_tensor[seed, 130304: 167168]
    eval_w11 = tf.reshape(eval_w11, [3, 3, 64, 64])
    eval_b11 = input_tensor[seed, 167168: 167232]

    eval_w12 = input_tensor[seed, 167232: 204096]
    eval_w12 = tf.reshape(eval_w12, [3, 3, 64, 64])
    eval_b12 = input_tensor[seed, 204096: 204160]

    eval_w13 = input_tensor[seed, 204160: 241024]
    eval_w13 = tf.reshape(eval_w13, [3, 3, 64, 64])
    eval_b13 = input_tensor[seed, 241024: 241088]

    eval_w14 = input_tensor[seed, 241088: 277952]
    eval_w14 = tf.reshape(eval_w14, [3, 3, 64, 64])
    eval_b14 = input_tensor[seed, 277952: 278016]

    eval_w15 = input_tensor[seed, 278016: 314880]
    eval_w15 = tf.reshape(eval_w15, [3, 3, 64, 64])
    eval_b15 = input_tensor[seed, 314880: 314944]

    eval_w16 = input_tensor[seed, 314944: 351808]
    eval_w16 = tf.reshape(eval_w16, [3, 3, 64, 64])
    eval_b16 = input_tensor[seed, 351808: 351872]


    # uint-3
    decoder_conv_w17 = y17_hat[seed, 0: 73728]
    decoder_conv_w17 = tf.reshape(decoder_conv_w17, [3, 3, 64, 128])
    decoder_conv_b17 = y17_hat[seed, 73728: 73856]

    decoder_conv_w18 = y18_hat[seed, 0: 73728]
    decoder_conv_w18 = tf.reshape(decoder_conv_w18, [3, 3, 64, 128])
    decoder_conv_b18 = y18_hat[seed, 73728: 73856]

    decoder_conv_w19 = y19_hat[seed, 0: 147456]
    decoder_conv_w19 = tf.reshape(decoder_conv_w19, [3, 3, 128, 128])
    decoder_conv_b19 = y19_hat[seed, 147456: 147584]

    decoder_conv_w20 = y20_hat[seed, 0: 147456]
    decoder_conv_w20 = tf.reshape(decoder_conv_w20, [3, 3, 128, 128])
    decoder_conv_b20 = y20_hat[seed, 147456: 147584]

    decoder_conv_w21 = y21_hat[seed, 0: 147456]
    decoder_conv_w21 = tf.reshape(decoder_conv_w21, [3, 3, 128, 128])
    decoder_conv_b21 = y21_hat[seed, 147456: 147584]

    decoder_conv_w22 = y22_hat[seed, 0: 147456]
    decoder_conv_w22 = tf.reshape(decoder_conv_w22, [3, 3, 128, 128])
    decoder_conv_b22 = y22_hat[seed, 147456: 147584]

    decoder_conv_w23 = y23_hat[seed, 0: 147456]
    decoder_conv_w23 = tf.reshape(decoder_conv_w23, [3, 3, 128, 128])
    decoder_conv_b23 = y23_hat[seed, 147456: 147584]

    decoder_conv_w24 = y24_hat[seed, 0: 147456]
    decoder_conv_w24 = tf.reshape(decoder_conv_w24, [3, 3, 128, 128])
    decoder_conv_b24 = y24_hat[seed, 147456: 147584]

    decoder_conv_w25 = y25_hat[seed, 0: 147456]
    decoder_conv_w25 = tf.reshape(decoder_conv_w25, [3, 3, 128, 128])
    decoder_conv_b25 = y25_hat[seed, 147456: 147584]

    decoder_conv_w26 = y26_hat[seed, 0: 147456]
    decoder_conv_w26 = tf.reshape(decoder_conv_w26, [3, 3, 128, 128])
    decoder_conv_b26 = y26_hat[seed, 147456: 147584]

    decoder_conv_w27 = y27_hat[seed, 0: 147456]
    decoder_conv_w27 = tf.reshape(decoder_conv_w27, [3, 3, 128, 128])
    decoder_conv_b27 = y27_hat[seed, 147456: 147584]

    decoder_conv_w28 = y28_hat[seed, 0: 147456]
    decoder_conv_w28 = tf.reshape(decoder_conv_w28, [3, 3, 128, 128])
    decoder_conv_b28 = y28_hat[seed, 147456: 147584]

    decoder_conv_w29 = y29_hat[seed, 0: 147456]
    decoder_conv_w29 = tf.reshape(decoder_conv_w29, [3, 3, 128, 128])
    decoder_conv_b29 = y29_hat[seed, 147456: 147584]

    # uint-4
    decoder_conv_w30 = y30_hat[seed, 0: 294912]
    decoder_conv_w30 = tf.reshape(decoder_conv_w30, [3, 3, 128, 256])
    decoder_conv_b30 = y30_hat[seed, 294912: 295168]

    decoder_conv_w31 = y31_hat[seed, 0: 294912]
    decoder_conv_w31 = tf.reshape(decoder_conv_w31, [3, 3, 128, 256])
    decoder_conv_b31 = y31_hat[seed, 294912: 295168]

    decoder_conv_w32 = y32_hat[seed, 0: 589824]
    decoder_conv_w32 = tf.reshape(decoder_conv_w32, [3, 3, 256, 256])
    decoder_conv_b32 = y32_hat[seed, 589824: 590080]

    decoder_conv_w33 = y33_hat[seed, 0: 589824]
    decoder_conv_w33 = tf.reshape(decoder_conv_w33, [3, 3, 256, 256])
    decoder_conv_b33 = y33_hat[seed, 589824: 590080]

    decoder_conv_w34 = y34_hat[seed, 0: 589824]
    decoder_conv_w34 = tf.reshape(decoder_conv_w34, [3, 3, 256, 256])
    decoder_conv_b34 = y34_hat[seed, 589824: 590080]

    decoder_conv_w35 = y35_hat[seed, 0: 589824]
    decoder_conv_w35 = tf.reshape(decoder_conv_w35, [3, 3, 256, 256])
    decoder_conv_b35 = y35_hat[seed, 589824: 590080]

    decoder_conv_w36 = y36_hat[seed, 0: 589824]
    decoder_conv_w36 = tf.reshape(decoder_conv_w36, [3, 3, 256, 256])
    decoder_conv_b36 = y36_hat[seed, 589824: 590080]

    decoder_fcn_w37 = y37_hat[seed, 0: 25600]
    decoder_fcn_w37 = tf.reshape(decoder_fcn_w37, [256, 100])
    decoder_fcn_b37 = y37_hat[seed, 25600: 25700]

    # 2-evaluate network
    # evaluate CIFAR10 images with the predicted weight parameters

    # init-conv
    eval_conv1 = tf.nn.conv2d(input_images, eval_w1, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv1 = tf.nn.bias_add(eval_conv1, eval_b1)
    eval_relu1 = tf.nn.relu(eval_conv1)

    eval_pool1 = tf.nn.max_pool(eval_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # unit-1
    eval_conv2 = tf.nn.conv2d(eval_pool1, eval_w2, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv2 = tf.nn.bias_add(eval_conv2, eval_b2)
    eval_relu2 = tf.nn.relu(eval_conv2)

    eval_conv3 = tf.nn.conv2d(eval_relu2, eval_w3, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv3 = tf.nn.bias_add(eval_conv3, eval_b3)
    eval_relu3 = tf.nn.relu(eval_conv3+eval_pool1)

    eval_conv4 = tf.nn.conv2d(eval_relu3, eval_w4, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv4 = tf.nn.bias_add(eval_conv4, eval_b4)
    eval_relu4 = tf.nn.relu(eval_conv4)

    eval_conv5 = tf.nn.conv2d(eval_relu4, eval_w5, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv5 = tf.nn.bias_add(eval_conv5, eval_b5)
    eval_relu5 = tf.nn.relu(eval_conv5+eval_relu3)

    eval_conv6 = tf.nn.conv2d(eval_relu5, eval_w6, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv6 = tf.nn.bias_add(eval_conv6, eval_b6)
    eval_relu6 = tf.nn.relu(eval_conv6)

    eval_conv7 = tf.nn.conv2d(eval_relu6, eval_w7, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv7 = tf.nn.bias_add(eval_conv7, eval_b7)
    eval_relu7 = tf.nn.relu(eval_conv7 + eval_relu5)

    # uint-2
    # shortcut
    eval_conv8 = tf.nn.conv2d(eval_relu7, eval_w8, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv8 = tf.nn.bias_add(eval_conv8, eval_b8)

    # block1
    eval_conv9 = tf.nn.conv2d(eval_relu7, eval_w9, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv9 = tf.nn.bias_add(eval_conv9, eval_b9)
    eval_relu9 = tf.nn.relu(eval_conv9)

    eval_conv10 = tf.nn.conv2d(eval_relu9, eval_w10, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv10 = tf.nn.bias_add(eval_conv10, eval_b10)
    eval_relu10 = tf.nn.relu(eval_conv10+eval_conv8)

    # block2
    eval_conv11 = tf.nn.conv2d(eval_relu10, eval_w11, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv11 = tf.nn.bias_add(eval_conv11, eval_b11)
    eval_relu11 = tf.nn.relu(eval_conv11)

    eval_conv12 = tf.nn.conv2d(eval_relu11, eval_w12, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv12 = tf.nn.bias_add(eval_conv12, eval_b12)
    eval_relu12 = tf.nn.relu(eval_conv12+eval_relu10)

    # block3
    eval_conv13 = tf.nn.conv2d(eval_relu12, eval_w13, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv13 = tf.nn.bias_add(eval_conv13, eval_b13)
    eval_relu13 = tf.nn.relu(eval_conv13)

    eval_conv14 = tf.nn.conv2d(eval_relu13, eval_w14, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv14 = tf.nn.bias_add(eval_conv14, eval_b14)
    eval_relu14 = tf.nn.relu(eval_conv14 + eval_relu12)

    # block4
    eval_conv15 = tf.nn.conv2d(eval_relu14, eval_w15, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv15 = tf.nn.bias_add(eval_conv15, eval_b15)
    eval_relu15 = tf.nn.relu(eval_conv15)

    eval_conv16 = tf.nn.conv2d(eval_relu15, eval_w16, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv16 = tf.nn.bias_add(eval_conv16, eval_b16)
    eval_relu16 = tf.nn.relu(eval_conv16 + eval_relu14)

    # uint-3
    # shortcut
    eval_conv17 = tf.nn.conv2d(eval_relu16, decoder_conv_w17, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv17 = tf.nn.bias_add(eval_conv17, decoder_conv_b17)
    # block1
    eval_conv18 = tf.nn.conv2d(eval_relu16, decoder_conv_w18, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv18 = tf.nn.bias_add(eval_conv18, decoder_conv_b18)
    eval_relu18 = tf.nn.relu(eval_conv18)

    eval_conv19 = tf.nn.conv2d(eval_relu18, decoder_conv_w19, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv19 = tf.nn.bias_add(eval_conv19, decoder_conv_b19)
    eval_relu19 = tf.nn.relu(eval_conv19+eval_conv17)

    # block2
    eval_conv20 = tf.nn.conv2d(eval_relu19, decoder_conv_w20, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv20 = tf.nn.bias_add(eval_conv20, decoder_conv_b20)
    eval_relu20 = tf.nn.relu(eval_conv20)

    eval_conv21 = tf.nn.conv2d(eval_relu20, decoder_conv_w21, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv21 = tf.nn.bias_add(eval_conv21, decoder_conv_b21)
    eval_relu21 = tf.nn.relu(eval_conv21+eval_relu19)

    # block3
    eval_conv22 = tf.nn.conv2d(eval_relu21, decoder_conv_w22, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv22 = tf.nn.bias_add(eval_conv22, decoder_conv_b22)
    eval_relu22 = tf.nn.relu(eval_conv22)

    eval_conv23 = tf.nn.conv2d(eval_relu22, decoder_conv_w23, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv23 = tf.nn.bias_add(eval_conv23, decoder_conv_b23)
    eval_relu23 = tf.nn.relu(eval_conv23 + eval_relu21)

    # block4
    eval_conv24 = tf.nn.conv2d(eval_relu23, decoder_conv_w24, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv24 = tf.nn.bias_add(eval_conv24, decoder_conv_b24)
    eval_relu24 = tf.nn.relu(eval_conv24)

    eval_conv25 = tf.nn.conv2d(eval_relu24, decoder_conv_w25, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv25 = tf.nn.bias_add(eval_conv25, decoder_conv_b25)
    eval_relu25 = tf.nn.relu(eval_conv21 + eval_relu23)

    # block5
    eval_conv26 = tf.nn.conv2d(eval_relu25, decoder_conv_w26, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv26 = tf.nn.bias_add(eval_conv26, decoder_conv_b26)
    eval_relu26 = tf.nn.relu(eval_conv26)

    eval_conv27 = tf.nn.conv2d(eval_relu26, decoder_conv_w27, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv27 = tf.nn.bias_add(eval_conv27, decoder_conv_b27)
    eval_relu27 = tf.nn.relu(eval_conv27 + eval_relu25)

    # block6
    eval_conv28 = tf.nn.conv2d(eval_relu27, decoder_conv_w28, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv28 = tf.nn.bias_add(eval_conv28, decoder_conv_b28)
    eval_relu28 = tf.nn.relu(eval_conv28)

    eval_conv29 = tf.nn.conv2d(eval_relu28, decoder_conv_w29, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv29 = tf.nn.bias_add(eval_conv29, decoder_conv_b29)
    eval_relu29 = tf.nn.relu(eval_conv29 + eval_relu27)

    # uint-4
    # shortcut
    eval_conv30 = tf.nn.conv2d(eval_relu29, decoder_conv_w30, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv30 = tf.nn.bias_add(eval_conv30, decoder_conv_b30)
    # block1
    eval_conv31 = tf.nn.conv2d(eval_relu29, decoder_conv_w31, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv31 = tf.nn.bias_add(eval_conv31, decoder_conv_b31)
    eval_relu31 = tf.nn.relu(eval_conv31)

    eval_conv32 = tf.nn.conv2d(eval_relu31, decoder_conv_w32, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv32 = tf.nn.bias_add(eval_conv32, decoder_conv_b32)
    eval_relu32 = tf.nn.relu(eval_conv32+eval_conv30)

    # block2
    eval_conv33 = tf.nn.conv2d(eval_relu32, decoder_conv_w33, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv33 = tf.nn.bias_add(eval_conv33, decoder_conv_b33)
    eval_relu33 = tf.nn.relu(eval_conv33)

    eval_conv34 = tf.nn.conv2d(eval_relu33, decoder_conv_w34, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv34 = tf.nn.bias_add(eval_conv34, decoder_conv_b34)
    eval_relu34 = tf.nn.relu(eval_conv34+eval_relu32)

    # block3
    eval_conv35 = tf.nn.conv2d(eval_relu34, decoder_conv_w35, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv35 = tf.nn.bias_add(eval_conv35, decoder_conv_b35)
    eval_relu35 = tf.nn.relu(eval_conv35)

    eval_conv36 = tf.nn.conv2d(eval_relu35, decoder_conv_w36, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv36 = tf.nn.bias_add(eval_conv36, decoder_conv_b36)
    eval_relu36 = tf.nn.relu(eval_conv36 + eval_relu34)

    eval_pool2 = tf.nn.avg_pool(eval_relu36, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print(eval_pool2.shape)
    p = tf.contrib.layers.flatten(eval_pool2)

    logits = tf.matmul(p, decoder_fcn_w37) + decoder_fcn_b37


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

    input_paras, nums_xs_batch = model_parameters_dataset(batch_size=200)

    predict_input, eval_input, eval_ya, eval_yb, lam_tensor, random_seed = create_placeholder()
    # 加入正则化
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZERATION)
    logits, istrain = forward_propagation(predict_input, eval_input, random_seed)

    variables = tf.trainable_variables()

    logits, loss, optimizer = compute_cost(logits, eval_ya, eval_yb, lam_tensor, 0.001, variables)

    accuracy = compute_accuracy(logits, eval_ya)

    init = tf.global_variables_initializer()
    # saver1 = tf.train.Saver(untrained_vars)
    saver2 = tf.train.Saver()

    train_xs, train_ys, test_xs, test_ys = load_dataset()
    nums_image_batch = int(train_xs.shape[0] / 125)
    l = list(np.arange(start=0, stop=nums_image_batch, step=1))

    costs = []
    test_accs = []

    max_acc = 0
    iterations = 0

    with tf.Session() as sess:
        sess.run(init)
        # saver1.restore(sess, './model2-vgg11/pred_model_for_vgg11/model.ckpt')
        for epoch in range(1, epoch_nums+1):
            epoch_cost = 0
            sum_of_time = 0
            temp = []
            for xs in input_paras:
                start = time.time()
                seed = np.random.randint(0, 200)
                X = xs.reshape([-1, 8, 8, 5498])
                iterations_cost = 0
                iterations += 1
                mini_batches = random_mini_batches(train_xs, train_ys, 125, seed=epoch)
                random_list = random.sample(l, 40)
                for index in random_list:
                    (mini_batch_X, mini_batch_Y) = mini_batches[index]
                    # mixup : 混合增强
                    x, mix_x, label_a, label_b, lam = mixup_data(mini_batch_X, mini_batch_Y, alpha=1)
                    _, cost, train_acc = sess.run([optimizer, loss, accuracy],
                                                  feed_dict={predict_input: X, eval_input: mix_x,
                                                             eval_ya: label_a, eval_yb: label_b,
                                                             lam_tensor: lam, random_seed: seed
                                                             })
                    iterations_cost += cost/40
                end = time.time()
                sum_of_time += (end - start)
                test_acc = sess.run(accuracy, feed_dict={predict_input: X, eval_input: test_xs, eval_ya: test_ys,
                                                         random_seed: seed})
                temp.append(test_acc)

                if test_acc > max_acc:
                    max_acc = test_acc
                    saver2.save(sess, "./save_model/model.ckpt")    # 保存精度最高时候的模型
                if iterations % 1 == 0:
                    print("Epoch {}/{}, Iteration {}/{}, Training cost is {}, Test accuracy is {}"
                          .format(epoch, epoch_nums, iterations, epoch_nums * nums_xs_batch, iterations_cost, test_acc))
                epoch_cost += iterations_cost/nums_xs_batch

            test_accs.append(max(temp))
            costs.append(epoch_cost)
            if epoch % 1 == 0:
                print("After {} Epoch, epoch cost is {}, Max test accuracy is {}, Test accuracy is {}, {} sec/batch"
                      .format(epoch, epoch_cost, max_acc, test_accs[-1], sum_of_time//nums_xs_batch))
        # 保存收敛时的模型
        # saver.save(sess, './save_model/model.ckpt')

    with open('./results/exp1/cost', 'wb') as f:
        f.write(pk.dumps(costs))
    with open('./results/exp1/test_acc', 'wb') as f:
        f.write(pk.dumps(test_accs))


if __name__=='__main__':
    main(epoch_nums=5)

