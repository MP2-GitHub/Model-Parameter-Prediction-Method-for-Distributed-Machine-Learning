import numpy as np
import tensorflow as tf
import pickle as pk


# 配置神经网络的参数
INPUT_NODE = 3072
OUTPUT_NODE = 10

# 输入图片的大小
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 64
CONV1_SIZE = 3

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 3

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 128
CONV3_SIZE = 3

# 第四层卷积层的尺寸和深度
CONV4_DEEP = 128
CONV4_SIZE = 3

# 第五层卷积层的尺寸和深度
CONV5_DEEP = 256
CONV5_SIZE = 3

# 第六层卷积层的尺寸和深度
CONV6_DEEP = 256
CONV6_SIZE = 3

# 第7层卷积层的尺寸和深度
CONV7_DEEP = 256
CONV7_SIZE = 3

# 第8层卷积层的尺寸和深度
CONV8_DEEP = 256
CONV8_SIZE = 3


# 初始化权重参数


def initialize_parameters():
    np.random.seed(2)

    w1 = np.random.randn(CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP) / np.sqrt(CONV1_SIZE * CONV1_SIZE * NUM_CHANNELS)
    w2 = np.random.randn(CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP) / np.sqrt(CONV2_SIZE * CONV2_SIZE * CONV1_DEEP)
    w3 = np.random.randn(CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP) / np.sqrt(CONV3_SIZE * CONV3_SIZE * CONV2_DEEP)
    w4 = np.random.randn(CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP) / np.sqrt(CONV4_SIZE * CONV4_SIZE * CONV3_DEEP)
    w5 = np.random.randn(CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP) / np.sqrt(CONV5_SIZE * CONV5_SIZE * CONV4_DEEP)
    w6 = np.random.randn(CONV6_SIZE, CONV6_SIZE, CONV5_DEEP, CONV6_DEEP) / np.sqrt(CONV6_SIZE * CONV6_SIZE * CONV5_DEEP)
    w7 = np.random.randn(CONV7_SIZE, CONV7_SIZE, CONV6_DEEP, CONV7_DEEP) / np.sqrt(CONV7_SIZE * CONV7_SIZE * CONV6_DEEP)
    w8 = np.random.randn(CONV8_SIZE, CONV8_SIZE, CONV7_DEEP, CONV8_DEEP) / np.sqrt(CONV8_SIZE * CONV8_SIZE * CONV7_DEEP)
    w9 = np.random.randn(1024, NUM_LABELS) / np.sqrt(1024)

    b1 = np.zeros((CONV1_DEEP,))
    b2 = np.zeros((CONV2_DEEP,))
    b3 = np.zeros((CONV3_DEEP,))
    b4 = np.zeros((CONV4_DEEP,))
    b5 = np.zeros((CONV5_DEEP,))
    b6 = np.zeros((CONV6_DEEP,))
    b7 = np.zeros((CONV7_DEEP,))
    b8 = np.zeros((CONV8_DEEP,))
    b9 = np.zeros((NUM_LABELS,))

    init_parameters = {"w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5, "w6": w6, "w7": w7, "w8": w8, "w9": w9,
                       "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "b6": b6, "b7": b7, "b8": b8, "b9": b9,
                       }

    return init_parameters


def create_placeholder():
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name="input_x")
    y_a = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y_a")
    y_b = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y_b")
    lam = tf.placeholder(tf.float32, name='lam')

    cv1_w = tf.placeholder(tf.float32, [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], name='cv1_w')
    cv1_b = tf.placeholder(tf.float32, [CONV1_DEEP], name='cv1_b')

    cv2_w = tf.placeholder(tf.float32, [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], name='cv2_w')
    cv2_b = tf.placeholder(tf.float32, [CONV2_DEEP], name='cv2_b')

    cv3_w = tf.placeholder(tf.float32, [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP], name='cv3_w')
    cv3_b = tf.placeholder(tf.float32, [CONV3_DEEP], name='cv3_b')

    cv4_w = tf.placeholder(tf.float32, [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP], name='cv4_w')
    cv4_b = tf.placeholder(tf.float32, [CONV4_DEEP], name='cv4_b')

    cv5_w = tf.placeholder(tf.float32, [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP], name='cv5_w')
    cv5_b = tf.placeholder(tf.float32, [CONV5_DEEP], name='cv5_b')

    cv6_w = tf.placeholder(tf.float32, [CONV6_SIZE, CONV6_SIZE, CONV5_DEEP, CONV6_DEEP], name='cv6_w')
    cv6_b = tf.placeholder(tf.float32, [CONV6_DEEP], name='cv6_b')

    cv7_w = tf.placeholder(tf.float32, [CONV7_SIZE, CONV7_SIZE, CONV6_DEEP, CONV7_DEEP], name='cv7_w')
    cv7_b = tf.placeholder(tf.float32, [CONV7_DEEP], name='cv7_b')

    cv8_w = tf.placeholder(tf.float32, [CONV8_SIZE, CONV8_SIZE, CONV7_DEEP, CONV8_DEEP], name='cv8_w')
    cv8_b = tf.placeholder(tf.float32, [CONV8_DEEP], name='cv8_b')

    fc_w = tf.placeholder(tf.float32, [1024, NUM_LABELS], name='fc_w')
    fc_b = tf.placeholder(tf.float32, [NUM_LABELS], name='fc_b')

    parameters = (
        cv1_w, cv1_b, cv2_w, cv2_b, cv3_w, cv3_b,
        cv4_w, cv4_b, cv5_w, cv5_b, cv6_w, cv6_b,
        cv7_w, cv7_b, cv8_w, cv8_b, fc_w, fc_b)

    return x, y_a, y_b, parameters, lam
