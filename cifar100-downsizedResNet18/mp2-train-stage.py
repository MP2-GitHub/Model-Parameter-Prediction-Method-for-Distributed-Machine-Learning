"""
code: MP2-train-prediction-model

"""

import time
import tensorflow as tf
import pickle as pk
import numpy as np
import math
from mixup import mixup_data
from seblock import SE_block
import random
import utils
from load_cifar100 import Cifar

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

    # cifar10 = Cifar10(path=r"./cifar-10-batches-py", one_hot=True)
    # cifar10._load_data()

    path = './cifar-100-python/'
    #
    cifar100 = Cifar(mode='cifar100', classes=100, path=path, one_hot=True)
    cifar100.load_cifar100()
    #
    print('training set:', cifar100.train.images.shape)
    print('training labels:', cifar100.train.labels.shape)
    print('test set:', cifar100.test.images.shape)
    print('test labels:', cifar100.test.labels.shape)

    # 准备训练集
    train_xs = cifar100.train.images / 255.0
    train_labels = cifar100.train.labels

    # # 准备测试集
    # np.random.seed(1)
    # permutation = list(np.random.permutation(10000))
    # shuffled_tx = cifar100.test.images[permutation, :, :, :] / 255.0
    # shuffled_ty = cifar100.test.labels[permutation, :]
    #
    # test_feeds = []
    # for k in range(10):
    #     test_xs = shuffled_tx[k * 1000:k * 1000 + 1000, :, :, :]
    #     test_labels = shuffled_ty[k * 1000:k * 1000 + 1000, :]
    #
    #     test_feed = (test_xs, test_labels)
    #
    #     test_feeds.append(test_feed)

    # 准备测试集
    test_xs = cifar100.test.images / 255.0
    test_labels = cifar100.test.labels
    print(test_xs.shape)

    return test_xs, test_labels

def create_placeholder():

    # weights samples
    x = tf.placeholder(tf.float32, [None, 8, 8, 2901], name='input_x')

    # uint-2
    # y6 = tf.placeholder(tf.float32, [None, 18496], name='label_6')
    # y7 = tf.placeholder(tf.float32, [None, 18496], name='label_7')
    # y8 = tf.placeholder(tf.float32, [None, 36928], name='label_8')
    # y9 = tf.placeholder(tf.float32, [None, 36928], name='label_9')
    # y10 = tf.placeholder(tf.float32, [None, 36928], name='label_10')

    # uint-3
    # y11 = tf.placeholder(tf.float32, [None, 73856], name='label_11')
    # y12 = tf.placeholder(tf.float32, [None, 73856], name='label_12')
    # y13 = tf.placeholder(tf.float32, [None, 147584], name='label_13')
    # y14 = tf.placeholder(tf.float32, [None, 147584], name='label_14')
    # y15 = tf.placeholder(tf.float32, [None, 147584], name='label_15')

    # uint-4
    # y16 = tf.placeholder(tf.float32, [None, 295168], name='label_16')
    # y17 = tf.placeholder(tf.float32, [None, 295168], name='label_17')
    # y18 = tf.placeholder(tf.float32, [None, 590080], name='label_18')
    # y19 = tf.placeholder(tf.float32, [None, 590080], name='label_19')
    # y20 = tf.placeholder(tf.float32, [None, 590080], name='label_20')

    # fc-layer
    # y21 = tf.placeholder(tf.float32, [None, 2570], name='label_21')

    # images
    images_input = tf.placeholder(tf.float32, [None, 32, 32, 3], name='images_input')
    images_label_a = tf.placeholder(tf.float32, [None, 100], name='y_a')
    images_label_b = tf.placeholder(tf.float32, [None, 100], name='y_b')
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
        w1 = tf.get_variable('weight', [3, 3, 2901, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
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
        w4 = tf.get_variable('fcn-weight-1', [flatten.shape[1], 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b4 = tf.get_variable('fcn-biases-1', [64], initializer=tf.constant_initializer(0.1))
        z4 = tf.matmul(flatten, w4) + b4
        # z4 = tf.layers.batch_normalization(z4, training=is_train2)
        a4 = tf.nn.relu(z4)
        a4 = tf.reshape(a4, [-1, 8, 8, 1])

        # uint-2
        # w6 = tf.get_variable('conv6-weight', [1, 1, 1, 289], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # z6 = tf.nn.conv2d(a4, w6, strides=[1, 1, 1, 1], padding='VALID')
        # y6_hat = tf.reshape(z6, [-1, 18496])
        #
        # w7 = tf.get_variable('conv7-weight', [1, 1, 1, 289], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # z7 = tf.nn.conv2d(a4, w7, strides=[1, 1, 1, 1], padding='VALID')
        # y7_hat = tf.reshape(z7, [-1, 18496])
        #
        # w8 = tf.get_variable('conv8-weight', [1, 1, 1, 577], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # z8 = tf.nn.conv2d(a4, w8, strides=[1, 1, 1, 1], padding='VALID')
        # y8_hat = tf.reshape(z8, [-1, 36928])
        #
        # w9 = tf.get_variable('conv9-weight', [1, 1, 1, 577], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # z9 = tf.nn.conv2d(a4, w9, strides=[1, 1, 1, 1], padding='VALID')
        # y9_hat = tf.reshape(z9, [-1, 36928])
        #
        # w10 = tf.get_variable('conv10-weight', [1, 1, 1,577], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # z10 = tf.nn.conv2d(a4, w10, strides=[1, 1, 1, 1], padding='VALID')
        # y10_hat = tf.reshape(z10, [-1, 36928])

        # uint-3
        w11 = tf.get_variable('conv11-weight', [1, 1, 1, 1154], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z11 = tf.nn.conv2d(a4, w11, strides=[1, 1, 1, 1], padding='VALID')
        y11_hat = tf.reshape(z11, [-1, 73856])

        w12 = tf.get_variable('conv12-weight', [1, 1, 1, 1154], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z12 = tf.nn.conv2d(a4, w12, strides=[1, 1, 1, 1], padding='VALID')
        y12_hat = tf.reshape(z12, [-1, 73856])

        w13 = tf.get_variable('conv13-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z13 = tf.nn.conv2d(a4, w13, strides=[1, 1, 1, 1], padding='VALID')
        y13_hat = tf.reshape(z13, [-1, 147584])

        w14 = tf.get_variable('conv14-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z14 = tf.nn.conv2d(a4, w14, strides=[1, 1, 1, 1], padding='VALID')
        y14_hat = tf.reshape(z14, [-1, 147584])

        w15 = tf.get_variable('conv15-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z15 = tf.nn.conv2d(a4, w15, strides=[1, 1, 1, 1], padding='VALID')
        y15_hat = tf.reshape(z15, [-1, 147584])

        # uint-4
        w16 = tf.get_variable('conv16-weight', [1, 1, 1, 4612], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z16 = tf.nn.conv2d(a4, w16, strides=[1, 1, 1, 1], padding='VALID')
        y16_hat = tf.reshape(z16, [-1, 295168])

        w17 = tf.get_variable('conv17-weight', [1, 1, 1, 4612], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z17 = tf.nn.conv2d(a4, w17, strides=[1, 1, 1, 1], padding='VALID')
        y17_hat = tf.reshape(z17, [-1, 295168])

        w18 = tf.get_variable('conv18-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z18 = tf.nn.conv2d(a4, w18, strides=[1, 1, 1, 1], padding='VALID')
        y18_hat = tf.reshape(z18, [-1, 590080])

        w19 = tf.get_variable('conv19-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z19 = tf.nn.conv2d(a4, w19, strides=[1, 1, 1, 1], padding='VALID')
        y19_hat = tf.reshape(z19, [-1, 590080])

        w20 = tf.get_variable('conv20-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z20 = tf.nn.conv2d(a4, w20, strides=[1, 1, 1, 1], padding='VALID')
        y20_hat = tf.reshape(z20, [-1, 590080])


    # Decoder-2: decode vgg11 fcn layer
    with tf.variable_scope('Decoder-2'):
        w21 = tf.get_variable('fcn1-weight', [flatten.shape[1], 25700], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z21 = tf.matmul(flatten, w21)
        y21_hat = tf.reshape(z21, [-1, 25700])

    # 输入切片
    # input_tensor = tf.reduce_mean(input_tensor, axis=0, keep_dims=True)
    input_tensor = tf.reshape(input_tensor, [-1, 185664])

    eval_w1 = input_tensor[seed, 0: 864]
    eval_w1 = tf.reshape(eval_w1, [3, 3, 3, 32])
    eval_b1 = input_tensor[seed, 864: 896]

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

    eval_w6 = input_tensor[seed, 37888: 56320]
    eval_w6 = tf.reshape(eval_w6, [3, 3, 32, 64])
    eval_b6 = input_tensor[seed, 56320: 56384]

    eval_w7 = input_tensor[seed, 56384: 74816]
    eval_w7 = tf.reshape(eval_w7, [3, 3, 32, 64])
    eval_b7 = input_tensor[seed, 74816: 74880]

    eval_w8 = input_tensor[seed, 74880: 111744]
    eval_w8 = tf.reshape(eval_w8, [3, 3, 64, 64])
    eval_b8 = input_tensor[seed, 111744: 111808]

    eval_w9 = input_tensor[seed, 111808: 148672]
    eval_w9 = tf.reshape(eval_w9, [3, 3, 64, 64])
    eval_b9 = input_tensor[seed, 148672: 148736]

    eval_w10 = input_tensor[seed, 148736: 185600]
    eval_w10 = tf.reshape(eval_w10, [3, 3, 64, 64])
    eval_b10 = input_tensor[seed, 185600: 185664]

    # eval_w11 = input_tensor[seed, 185664: 259392]
    # eval_w11 = tf.reshape(eval_w11, [3, 3, 64, 128])
    # eval_b11 = input_tensor[seed, 259392: 259520]
    #
    # eval_w12 = input_tensor[seed, 259520: 333248]
    # eval_w12 = tf.reshape(eval_w12, [3, 3, 64, 128])
    # eval_b12 = input_tensor[seed, 333248: 333376]

    # Decoder 切片
    # 张量切片: biases = z5[1, 30720:30840]
    # uint-2
    # decoder_conv_w6 = y6_hat[seed, 0: 18432]
    # decoder_conv_w6 = tf.reshape(decoder_conv_w6, [3, 3, 32, 64])
    # decoder_conv_b6 = y6_hat[seed, 18432: 18496]
    #
    # decoder_conv_w7 = y7_hat[seed, 0: 18432]
    # decoder_conv_w7 = tf.reshape(decoder_conv_w7, [3, 3, 32, 64])
    # decoder_conv_b7 = y7_hat[seed, 18432: 18496]
    #
    # decoder_conv_w8 = y8_hat[seed, 0: 36864]
    # decoder_conv_w8 = tf.reshape(decoder_conv_w8, [3, 3, 64, 64])
    # decoder_conv_b8 = y8_hat[seed, 36864: 36928]
    #
    # decoder_conv_w9 = y9_hat[seed, 0: 36864]
    # decoder_conv_w9 = tf.reshape(decoder_conv_w9, [3, 3, 64, 64])
    # decoder_conv_b9 = y9_hat[seed, 36864: 36928]
    #
    # decoder_conv_w10 = y10_hat[seed, 0: 36864]
    # decoder_conv_w10 = tf.reshape(decoder_conv_w10, [3, 3, 64, 64])
    # decoder_conv_b10 = y10_hat[seed, 36864: 36928]

    # uint-3
    decoder_conv_w11 = y11_hat[seed, 0: 73728]
    decoder_conv_w11 = tf.reshape(decoder_conv_w11, [3, 3, 64, 128])
    decoder_conv_b11 = y11_hat[seed, 73728: 73856]

    decoder_conv_w12 = y12_hat[seed, 0: 73728]
    decoder_conv_w12 = tf.reshape(decoder_conv_w12, [3, 3, 64, 128])
    decoder_conv_b12 = y12_hat[seed, 73728: 73856]

    decoder_conv_w13 = y13_hat[seed, 0: 147456]
    decoder_conv_w13 = tf.reshape(decoder_conv_w13, [3, 3, 128, 128])
    decoder_conv_b13 = y13_hat[seed, 147456: 147584]

    decoder_conv_w14 = y14_hat[seed, 0: 147456]
    decoder_conv_w14 = tf.reshape(decoder_conv_w14, [3, 3, 128, 128])
    decoder_conv_b14 = y14_hat[seed, 147456: 147584]

    decoder_conv_w15 = y15_hat[seed, 0: 147456]
    decoder_conv_w15 = tf.reshape(decoder_conv_w15, [3, 3, 128, 128])
    decoder_conv_b15 = y15_hat[seed, 147456: 147584]

    # uint-4
    decoder_conv_w16 = y16_hat[seed, 0: 294912]
    decoder_conv_w16 = tf.reshape(decoder_conv_w16, [3, 3, 128, 256])
    decoder_conv_b16 = y16_hat[seed, 294912: 295168]

    decoder_conv_w17 = y17_hat[seed, 0: 294912]
    decoder_conv_w17 = tf.reshape(decoder_conv_w17, [3, 3, 128, 256])
    decoder_conv_b17 = y17_hat[seed, 294912: 295168]

    decoder_conv_w18 = y18_hat[seed, 0: 589824]
    decoder_conv_w18 = tf.reshape(decoder_conv_w18, [3, 3, 256, 256])
    decoder_conv_b18 = y18_hat[seed, 589824: 590080]

    decoder_conv_w19 = y19_hat[seed, 0: 589824]
    decoder_conv_w19 = tf.reshape(decoder_conv_w19, [3, 3, 256, 256])
    decoder_conv_b19 = y19_hat[seed, 589824: 590080]

    decoder_conv_w20 = y20_hat[seed, 0: 589824]
    decoder_conv_w20 = tf.reshape(decoder_conv_w20, [3, 3, 256, 256])
    decoder_conv_b20 = y20_hat[seed, 589824: 590080]

    decoder_fcn_w21 = y21_hat[seed, 0: 25600]
    decoder_fcn_w21 = tf.reshape(decoder_fcn_w21, [256, 100])
    decoder_fcn_b21 = y21_hat[seed, 25600: 25700]

    # 2-evaluate network
    # evaluate CIFAR10 images with the predicted weight parameters

    # init-conv
    eval_conv1 = tf.nn.conv2d(input_images, eval_w1, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv1 = tf.nn.bias_add(eval_conv1, eval_b1)
    eval_relu1 = tf.nn.relu(eval_conv1)

    eval_pool1 = tf.nn.max_pool(eval_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # uint-1
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

    # uint-2
    # shortcut
    eval_conv6 = tf.nn.conv2d(eval_relu5, eval_w6, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv6 = tf.nn.bias_add(eval_conv6, eval_b6)
    # block1
    eval_conv7 = tf.nn.conv2d(eval_relu5, eval_w7, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv7 = tf.nn.bias_add(eval_conv7, eval_b7)
    eval_relu7 = tf.nn.relu(eval_conv7)

    eval_conv8 = tf.nn.conv2d(eval_relu7, eval_w8, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv8 = tf.nn.bias_add(eval_conv8, eval_b8)
    eval_relu8 = tf.nn.relu(eval_conv8+eval_conv6)
    # block2
    eval_conv9 = tf.nn.conv2d(eval_relu8, eval_w9, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv9 = tf.nn.bias_add(eval_conv9, eval_b9)
    eval_relu9 = tf.nn.relu(eval_conv9)

    eval_conv10 = tf.nn.conv2d(eval_relu9, eval_w10, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv10 = tf.nn.bias_add(eval_conv10, eval_b10)
    eval_relu10 = tf.nn.relu(eval_conv10+eval_relu8)

    # uint-3
    # shortcut
    eval_conv11 = tf.nn.conv2d(eval_relu10, decoder_conv_w11, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv11 = tf.nn.bias_add(eval_conv11, decoder_conv_b11)
    # block1
    eval_conv12 = tf.nn.conv2d(eval_relu10, decoder_conv_w12, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv12 = tf.nn.bias_add(eval_conv12, decoder_conv_b12)
    eval_relu12 = tf.nn.relu(eval_conv12)

    eval_conv13 = tf.nn.conv2d(eval_relu12, decoder_conv_w13, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv13 = tf.nn.bias_add(eval_conv13, decoder_conv_b13)
    eval_relu13 = tf.nn.relu(eval_conv13+eval_conv11)
    # block2
    eval_conv14 = tf.nn.conv2d(eval_relu13, decoder_conv_w14, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv14 = tf.nn.bias_add(eval_conv14, decoder_conv_b14)
    eval_relu14 = tf.nn.relu(eval_conv14)

    eval_conv15 = tf.nn.conv2d(eval_relu14, decoder_conv_w15, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv15 = tf.nn.bias_add(eval_conv15, decoder_conv_b15)
    eval_relu15 = tf.nn.relu(eval_conv15+eval_relu13)

    # uint-4
    # shortcut
    eval_conv16 = tf.nn.conv2d(eval_relu15, decoder_conv_w16, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv16 = tf.nn.bias_add(eval_conv16, decoder_conv_b16)
    # block1
    eval_conv17 = tf.nn.conv2d(eval_relu15, decoder_conv_w17, strides=[1, 2, 2, 1], padding='SAME')
    eval_conv17 = tf.nn.bias_add(eval_conv17, decoder_conv_b17)
    eval_relu17 = tf.nn.relu(eval_conv17)

    eval_conv18 = tf.nn.conv2d(eval_relu17, decoder_conv_w18, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv18 = tf.nn.bias_add(eval_conv18, decoder_conv_b18)
    eval_relu18 = tf.nn.relu(eval_conv18+eval_conv16)
    # block2
    eval_conv19 = tf.nn.conv2d(eval_relu18, decoder_conv_w19, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv19 = tf.nn.bias_add(eval_conv19, decoder_conv_b19)
    eval_relu19 = tf.nn.relu(eval_conv19)

    eval_conv20 = tf.nn.conv2d(eval_relu19, decoder_conv_w20, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv20 = tf.nn.bias_add(eval_conv20, decoder_conv_b20)
    eval_relu20 = tf.nn.relu(eval_conv20+eval_relu18)

    eval_pool2 = tf.nn.avg_pool(eval_relu20, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print(eval_pool2.shape)
    p = tf.contrib.layers.flatten(eval_pool2)

    logits = tf.matmul(p, decoder_fcn_w21) + decoder_fcn_b21

    # y6_hat = y6_hat[seed, 0: 18496]
    # y6_hat = tf.reshape(y6_hat, [1, -1])
    #
    # y7_hat = y7_hat[seed, 0: 18496]
    # y7_hat = tf.reshape(y7_hat, [1, -1])
    #
    # y8_hat = y8_hat[seed, 0: 36928]
    # y8_hat = tf.reshape(y8_hat, [1, -1])
    #
    # y9_hat = y9_hat[seed, 0: 36928]
    # y9_hat = tf.reshape(y9_hat, [1, -1])
    #
    # y10_hat = y10_hat[seed, 0: 36928]
    # y10_hat = tf.reshape(y10_hat, [1, -1])

    # y11_hat = y11_hat[seed, 0: 73856]
    # y11_hat = tf.reshape(y11_hat, [1, -1])
    #
    # y12_hat = y12_hat[seed, 0: 73856]
    # y12_hat = tf.reshape(y12_hat, [1, -1])

    # y13_hat = y13_hat[seed, 0: 147584]
    # y13_hat = tf.reshape(y13_hat, [1, -1])
    #
    # y14_hat = y14_hat[seed, 0: 147584]
    # y14_hat = tf.reshape(y14_hat, [1, -1])
    #
    # y15_hat = y15_hat[seed, 0: 147584]
    # y15_hat = tf.reshape(y15_hat, [1, -1])
    #
    # y16_hat = y16_hat[seed, 0: 295168]
    # y16_hat = tf.reshape(y16_hat, [1, -1])
    #
    # y17_hat = y17_hat[seed, 0: 295168]
    # y17_hat = tf.reshape(y17_hat, [1, -1])
    #
    # y18_hat = y18_hat[seed, 0: 590080]
    # y18_hat = tf.reshape(y18_hat, [1, -1])
    #
    # y19_hat = y19_hat[seed, 0: 590080]
    # y19_hat = tf.reshape(y19_hat, [1, -1])
    #
    # y20_hat = y20_hat[seed, 0: 590080]
    # y20_hat = tf.reshape(y20_hat, [1, -1])
    #
    # y21_hat = y21_hat[seed, 0: 2570]
    # y21_hat = tf.reshape(y21_hat, [1, -1])

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

def model_parameters_dataset3(batch_size, index):

    with open('./model_parameters/xs_layers_12345_{}'.format(index), 'rb') as f:
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
    train_dataset = utils.load_data(path='./cifar-100-classed',)

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
                    X = xs.reshape([-1, 8, 8, 2901])
                    iterations_cost = 0
                    iterations += 1
                    # sample 10% images from each class
                    mini_batches, image_batches = utils.sample_image(train_dataset, 100, percent=0.25, seed=iterations, mini_batch_size=125)

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
