import numpy as np
import tensorflow as tf
import pickle as pk


# 配置神经网络的参数
INPUT_NODE = 3072
OUTPUT_NODE = 100

# 输入图片的大小
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_LABELS = 100

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

# 第9层卷积层的尺寸和深度
CONV9_DEEP = 256
CONV9_SIZE = 3

# 第10层卷积层的尺寸和深度
CONV10_DEEP = 256
CONV10_SIZE = 3

# 第11层卷积层的尺寸和深度
CONV11_DEEP = 256
CONV11_SIZE = 3

# 第12层卷积层的尺寸和深度
CONV12_DEEP = 256
CONV12_SIZE = 3

# 第13层卷积层的尺寸和深度
CONV13_DEEP = 256
CONV13_SIZE = 3

# 第14层全连接层的尺寸和深度
FC1_SIZE = 100
# 第15层全连接层的尺寸和深度
FC2_SIZE = 100

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
    w9 = np.random.randn(CONV9_SIZE, CONV9_SIZE, CONV8_DEEP, CONV9_DEEP) / np.sqrt(CONV9_SIZE * CONV9_SIZE * CONV8_DEEP)
    # w10 = np.random.randn(CONV10_SIZE, CONV10_SIZE, CONV9_DEEP, CONV10_DEEP) / np.sqrt(CONV10_SIZE * CONV10_SIZE * CONV9_DEEP)

    w10 = np.random.randn(CONV11_SIZE, CONV11_SIZE, CONV10_DEEP, CONV11_DEEP) / np.sqrt(CONV11_SIZE * CONV11_SIZE * CONV10_DEEP)
    w11 = np.random.randn(CONV12_SIZE, CONV12_SIZE, CONV11_DEEP, CONV12_DEEP) / np.sqrt(CONV12_SIZE * CONV12_SIZE * CONV11_DEEP)
    w12 = np.random.randn(CONV13_SIZE, CONV13_SIZE, CONV12_DEEP, CONV13_DEEP) / np.sqrt(CONV13_SIZE * CONV13_SIZE * CONV12_DEEP)

    w13 = np.random.randn(256, FC1_SIZE) / np.sqrt(256)
    w14 = np.random.randn(FC1_SIZE, FC2_SIZE) / np.sqrt(FC1_SIZE)
    w15 = np.random.randn(FC2_SIZE, NUM_LABELS) / np.sqrt(FC2_SIZE)

    b1 = np.zeros((CONV1_DEEP,))
    b2 = np.zeros((CONV2_DEEP,))
    b3 = np.zeros((CONV3_DEEP,))
    b4 = np.zeros((CONV4_DEEP,))
    b5 = np.zeros((CONV5_DEEP,))
    b6 = np.zeros((CONV6_DEEP,))
    b7 = np.zeros((CONV7_DEEP,))
    b8 = np.zeros((CONV8_DEEP,))
    b9 = np.zeros((CONV9_DEEP,))
    # b10 = np.zeros((CONV10_DEEP,))
    b10 = np.zeros((CONV11_DEEP,))
    b11 = np.zeros((CONV12_DEEP,))
    b12 = np.zeros((CONV13_DEEP,))

    b13 = np.zeros((FC1_SIZE,))
    b14 = np.zeros((FC2_SIZE,))
    b15 = np.zeros((NUM_LABELS,))

    init_parameters = {"w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5, "w6": w6, "w7": w7, "w8": w8, "w9": w9,
                       "w10": w10, "w11": w11, "w12": w12, "w13": w13, "w14": w14, "w15": w15,
                       "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "b6": b6, "b7": b7, "b8": b8, "b9": b9,
                       "b10": b10, "b11": b11, "b12": b12, "b13": b13, "b14": b14, "b15": b15
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

    cv9_w = tf.placeholder(tf.float32, [CONV9_SIZE, CONV9_SIZE, CONV8_DEEP, CONV9_DEEP], name='cv9_w')
    cv9_b = tf.placeholder(tf.float32, [CONV9_DEEP], name='cv9_b')

    # cv10_w = tf.placeholder(tf.float32, [CONV10_SIZE, CONV10_SIZE, CONV9_DEEP, CONV10_DEEP], name='cv10_w')
    # cv10_b = tf.placeholder(tf.float32, [CONV10_DEEP], name='cv10_b')

    cv10_w = tf.placeholder(tf.float32, [CONV11_SIZE, CONV11_SIZE, CONV10_DEEP, CONV11_DEEP], name='cv10_w')
    cv10_b = tf.placeholder(tf.float32, [CONV11_DEEP], name='cv10_b')

    cv11_w = tf.placeholder(tf.float32, [CONV12_SIZE, CONV12_SIZE, CONV11_DEEP, CONV12_DEEP], name='cv11_w')
    cv11_b = tf.placeholder(tf.float32, [CONV12_DEEP], name='cv11_b')

    cv12_w = tf.placeholder(tf.float32, [CONV13_SIZE, CONV13_SIZE, CONV12_DEEP, CONV13_DEEP], name='cv12_w')
    cv12_b = tf.placeholder(tf.float32, [CONV13_DEEP], name='cv12_b')

    fc13_w = tf.placeholder(tf.float32, [256, FC1_SIZE], name='fc1_w')
    fc13_b = tf.placeholder(tf.float32, [FC1_SIZE], name='fc1_b')

    fc14_w = tf.placeholder(tf.float32, [FC1_SIZE, FC2_SIZE], name='fc2_w')
    fc14_b = tf.placeholder(tf.float32, [FC2_SIZE], name='fc2_b')

    fc15_w = tf.placeholder(tf.float32, [FC2_SIZE, NUM_LABELS], name='fc3_w')
    fc15_b = tf.placeholder(tf.float32, [NUM_LABELS], name='fc3_b')

    parameters = (
        cv1_w, cv1_b, cv2_w, cv2_b, cv3_w, cv3_b,
        cv4_w, cv4_b, cv5_w, cv5_b, cv6_w, cv6_b,
        cv7_w, cv7_b, cv8_w, cv8_b, cv9_w, cv9_b,
        cv10_w, cv10_b, cv11_w, cv11_b, cv12_w, cv12_b,
        fc13_w, fc13_b, fc14_w, fc14_b, fc15_w, fc15_b)

    return x, y_a, y_b, parameters, lam
