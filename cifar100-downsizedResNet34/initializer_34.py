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

    # unit-1
    conv1_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv1_w')
    conv1_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv1_b')

    conv2_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv2_w')
    conv2_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv2_b')

    conv3_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv3_w')
    conv3_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv3_b')

    conv4_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv4_w')
    conv4_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv4_b')

    conv5_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv5_w')
    conv5_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv5_b')

    conv6_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv6_w')
    conv6_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv6_b')

    # unit-2
    shortcut1_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2], name='shortcut1_w')
    shortcut1_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='shortcut1_b')

    conv7_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2], name='conv7_w')
    conv7_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv7_b')

    conv8_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv8_w')
    conv8_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv8_b')

    conv9_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv9_w')
    conv9_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv9_b')

    conv10_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv10_w')
    conv10_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv10_b')

    conv11_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv11_w')
    conv11_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv11_b')

    conv12_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv12_w')
    conv12_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv12_b')

    conv13_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv13_w')
    conv13_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv13_b')

    conv14_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv14_w')
    conv14_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv14_b')

    # unit-3
    shortcut2_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3], name='shortcut3_w')
    shortcut2_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='shortcut3_b')

    conv15_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3], name='conv15_w')
    conv15_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv15_b')

    conv16_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv16_w')
    conv16_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv16_b')

    conv17_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv17_w')
    conv17_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv17_b')

    conv18_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv18_w')
    conv18_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv18_b')

    conv19_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv19_w')
    conv19_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv19_b')

    conv20_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv20_w')
    conv20_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv20_b')

    conv21_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv21_w')
    conv21_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv21_b')

    conv22_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv22_w')
    conv22_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv22_b')

    conv23_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv23_w')
    conv23_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv23_b')

    conv24_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv24_w')
    conv24_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv24_b')

    conv25_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv25_w')
    conv25_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv25_b')

    conv26_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv26_w')
    conv26_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv26_b')

    # unit-4
    shortcut3_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4], name='shortcut5_w')
    shortcut3_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='shortcut5_b')

    conv27_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4], name='conv27_w')
    conv27_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv27_b')

    conv28_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv28_w')
    conv28_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv28_b')

    conv29_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv29_w')
    conv29_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv29_b')

    conv30_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv30_w')
    conv30_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv30_b')

    conv31_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv31_w')
    conv31_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv31_b')

    conv32_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv32_w')
    conv32_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv32_b')

    fc_w = tf.placeholder(tf.float32, [CONV_DEEP4, OUTPUT_SIZE], name='fc_w')
    fc_b = tf.placeholder(tf.float32, [OUTPUT_SIZE], name='fc_b')

    parameters = (init_conv_w, init_conv_b, conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, conv4_w, conv4_b, conv5_w, conv5_b, conv6_w, conv6_b,
                  shortcut1_w, shortcut1_b, conv7_w, conv7_b, conv8_w, conv8_b, conv9_w, conv9_b, conv10_w, conv10_b,
                  conv11_w, conv11_b, conv12_w, conv12_b, conv13_w, conv13_b, conv14_w, conv14_b,
                  shortcut2_w, shortcut2_b, conv15_w, conv15_b, conv16_w, conv16_b, conv17_w, conv17_b, conv18_w, conv18_b, conv19_w, conv19_b,
                  conv20_w, conv20_b, conv21_w, conv21_b, conv22_w, conv22_b, conv23_w, conv23_b, conv24_w, conv24_b, conv25_w, conv25_b, conv26_w, conv26_b,
                  shortcut3_w, shortcut3_b, conv27_w, conv27_b, conv28_w, conv28_b, conv29_w, conv29_b, conv30_w, conv30_b, conv31_w, conv31_b, conv32_w, conv32_b,
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
    w6 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w7 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)

    b2 = np.zeros((CONV_DEEP1,))
    b3 = np.zeros((CONV_DEEP1,))
    b4 = np.zeros((CONV_DEEP1,))
    b5 = np.zeros((CONV_DEEP1,))
    b6 = np.zeros((CONV_DEEP1,))
    b7 = np.zeros((CONV_DEEP1,))

    # shortcut1
    w8 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    b8 = np.zeros((CONV_DEEP2,))

    # conv2
    w9 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w10 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w11 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w12 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w13 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w14 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w15 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w16 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)

    b9 = np.zeros((CONV_DEEP2,))
    b10 = np.zeros((CONV_DEEP2,))
    b11 = np.zeros((CONV_DEEP2,))
    b12 = np.zeros((CONV_DEEP2,))
    b13 = np.zeros((CONV_DEEP2,))
    b14 = np.zeros((CONV_DEEP2,))
    b15 = np.zeros((CONV_DEEP2,))
    b16 = np.zeros((CONV_DEEP2,))

    # shortcut2
    w17 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    b17 = np.zeros((CONV_DEEP3,))

    # conv3
    w18 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w19 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w20 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w21 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w22 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w23 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w24 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w25 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w26 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w27 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w28 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w29 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)

    b18 = np.zeros((CONV_DEEP3,))
    b19 = np.zeros((CONV_DEEP3,))
    b20 = np.zeros((CONV_DEEP3,))
    b21 = np.zeros((CONV_DEEP3,))
    b22 = np.zeros((CONV_DEEP3,))
    b23 = np.zeros((CONV_DEEP3,))
    b24 = np.zeros((CONV_DEEP3,))
    b25 = np.zeros((CONV_DEEP3,))
    b26 = np.zeros((CONV_DEEP3,))
    b27 = np.zeros((CONV_DEEP3,))
    b28 = np.zeros((CONV_DEEP3,))
    b29 = np.zeros((CONV_DEEP3,))

    # shortcut3
    w30 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    b30 = np.zeros((CONV_DEEP4,))

    # conv4
    w31 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w32 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)
    w33 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)
    w34 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)
    w35 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)
    w36 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)

    b31 = np.zeros((CONV_DEEP4,))
    b32 = np.zeros((CONV_DEEP4,))
    b33 = np.zeros((CONV_DEEP4,))
    b34 = np.zeros((CONV_DEEP4,))
    b35 = np.zeros((CONV_DEEP4,))
    b36 = np.zeros((CONV_DEEP4,))

    # fc_layer
    w37 = np.random.randn(CONV_DEEP4, OUTPUT_SIZE) / np.sqrt(CONV_DEEP4)
    b37 = np.zeros((OUTPUT_SIZE,))
    initial_parameter = {'w1': w1, 'b1': b1,
                         'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3, 'w4': w4, 'b4': b4, 'w5': w5, 'b5': b5, 'w6': w6, 'b6': b6, 'w7': w7, 'b7': b7,
                         'w8': w8, 'b8': b8,
                         'w9': w9, 'b9': b9, 'w10':w10, 'b10': b10, 'w11': w11, 'b11': b11, 'w12':w12, 'b12':b12,
                         'w13': w13, 'b13': b13, 'w14': w14, 'b14': b14, 'w15': w15, 'b15': b15, 'w16': w16, 'b16': b16,
                         'w17': w17, 'b17': b17,
                         'w18': w18, 'b18': b18, 'w19': w19, 'b19': b19, 'w20': w20, 'b20': b20, 'w21': w21, 'b21': b21,'w22': w22, 'b22': b22, 'w23': w23, 'b23': b23,
                         'w24': w24, 'b24': b24, 'w25': w25, 'b25': b25, 'w26': w26, 'b26': b26,'w27': w27, 'b27': b27, 'w28': w28, 'b28': b28, 'w29': w29, 'b29': b29,
                         'w30': w30, 'b30': b30,
                         'w31': w31, 'b31': b31, 'w32': w32, 'b32': b32, 'w33': w33, 'b33': b33, 'w34': w34, 'b34': b34, 'w35': w35, 'b35': b35,'w36': w36, 'b36': b36, 'w37': w37, 'b37': b37,
                         }
    return initial_parameter






