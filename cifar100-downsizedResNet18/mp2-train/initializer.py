import tensorflow as tf
import numpy as np

# 参数设置
IMAGE_SIZE = 32

CHANELS_SIZE = 3

OUTPUT_SIZE = 100

# 核大小
CONV_SIZE = 3

# 卷积核数
CONV_DEEP1 = 32
CONV_DEEP2 = 64
CONV_DEEP3 = 128
CONV_DEEP4 = 256


def create_placeholder():
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANELS_SIZE], name='input_x')
    y_a = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="input_y_a")
    y_b = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="input_y_b")
    lam_tensor = tf.placeholder(tf.float32, name='lam')

    is_train = tf.placeholder_with_default(False, (), 'is_train')

    # 模型占位符
    # init-conv
    init_conv_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CHANELS_SIZE, CONV_DEEP1], name='init_conv_w')
    init_conv_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='init_conv_b')

    # uint-1
    conv1_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv1_w')
    conv1_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv1_b')

    conv2_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv2_w')
    conv2_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv2_b')

    conv3_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv3_w')
    conv3_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv3_b')

    conv4_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv4_w')
    conv4_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv4_b')

    # uint-2
    shortcut1_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2], name='shortcut1_w')
    shortcut1_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='shortcut1_b')

    conv5_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2], name='conv5_w')
    conv5_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv5_b')

    conv6_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv6_w')
    conv6_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv6_b')

    conv7_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv7_w')
    conv7_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv7_b')

    conv8_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv8_w')
    conv8_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv8_b')

    # uint-3
    shortcut2_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3], name='shortcut3_w')
    shortcut2_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='shortcut3_b')

    conv9_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3], name='conv9_w')
    conv9_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv9_b')

    conv10_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv10_w')
    conv10_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv10_b')

    conv11_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv11_w')
    conv11_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv11_b')

    conv12_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv12_w')
    conv12_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv12_b')

    # uint-4
    shortcut3_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4], name='shortcut5_w')
    shortcut3_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='shortcut5_b')

    conv13_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4], name='conv13_w')
    conv13_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv13_b')

    conv14_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv14_w')
    conv14_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv14_b')

    conv15_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv15_w')
    conv15_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv15_b')

    conv16_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv16_w')
    conv16_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv16_b')

    fc_w = tf.placeholder(tf.float32, [CONV_DEEP4, OUTPUT_SIZE], name='fc_w')
    fc_b = tf.placeholder(tf.float32, [OUTPUT_SIZE], name='fc_b')

    parameters = (init_conv_w, init_conv_b, conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, conv4_w, conv4_b,
                  shortcut1_w, shortcut1_b, conv5_w, conv5_b, conv6_w, conv6_b, conv7_w, conv7_b, conv8_w, conv8_b,
                  shortcut2_w, shortcut2_b, conv9_w, conv9_b, conv10_w, conv10_b, conv11_w, conv11_b, conv12_w, conv12_b,
                  shortcut3_w, shortcut3_b, conv13_w, conv13_b, conv14_w, conv14_b, conv15_w, conv15_b, conv16_w, conv16_b,
                  fc_w, fc_b)

    return x, y_a, y_b, lam_tensor, parameters


def initial_parameters():

    np.random.seed(1)

    # init_conv
    w1 = np.random.randn(CONV_SIZE, CONV_SIZE, CHANELS_SIZE, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CHANELS_SIZE)
    b1 = np.zeros((CONV_DEEP1,))

    # conv1
    w2 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w3 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w4 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w5 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)

    b2 = np.zeros((CONV_DEEP1,))
    b3 = np.zeros((CONV_DEEP1,))
    b4 = np.zeros((CONV_DEEP1,))
    b5 = np.zeros((CONV_DEEP1,))

    # shortcut1
    w6 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    b6 = np.zeros((CONV_DEEP2,))

    # conv2
    w7 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w8 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w9 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w10 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)

    b7 = np.zeros((CONV_DEEP2,))
    b8 = np.zeros((CONV_DEEP2,))
    b9 = np.zeros((CONV_DEEP2,))
    b10 = np.zeros((CONV_DEEP2,))

    # shortcut2
    w11 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    b11 = np.zeros((CONV_DEEP3,))

    # conv3
    w12 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w13 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w14 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w15 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)

    b12 = np.zeros((CONV_DEEP3,))
    b13 = np.zeros((CONV_DEEP3,))
    b14 = np.zeros((CONV_DEEP3,))
    b15 = np.zeros((CONV_DEEP3,))

    # shortcut3
    w16 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    b16 = np.zeros((CONV_DEEP4,))

    # conv4
    w17 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w18 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)
    w19 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)
    w20 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)

    b17 = np.zeros((CONV_DEEP4,))
    b18 = np.zeros((CONV_DEEP4,))
    b19 = np.zeros((CONV_DEEP4,))
    b20 = np.zeros((CONV_DEEP4,))

    # fc_layer
    w21 = np.random.randn(CONV_DEEP4, OUTPUT_SIZE) / np.sqrt(CONV_DEEP4)
    b21 = np.zeros((OUTPUT_SIZE,))
    initial_parameter = {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2, 'w3':w3, 'b3':b3, 'w4':w4, 'b4':b4,
                         'w5':w5, 'b5':b5, 'w6':w6, 'b6':b6, 'w7':w7, 'b7':b7, 'w8':w8, 'b8':b8,
                         'w9':w9, 'b9':b9, 'w10':w10, 'b10':b10, 'w11':w11, 'b11':b11, 'w12':w12, 'b12':b12,
                         'w13': w13, 'b13': b13, 'w14': w14, 'b14': b14, 'w15': w15, 'b15': b15, 'w16': w16, 'b16': b16,
                         'w17': w17, 'b17': b17, 'w18': w18, 'b18': b18, 'w19': w19, 'b19': b19,
                         'w20': w20, 'b20': b20, 'w21': w21, 'b21': b21
                         }

    return initial_parameter




