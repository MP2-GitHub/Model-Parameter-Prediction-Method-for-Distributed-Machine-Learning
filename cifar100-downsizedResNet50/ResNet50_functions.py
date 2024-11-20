from socket import *
from queue import Queue
import pickle as pk
import tensorflow as tf
from seblock import SE_block


# 建立TCP连接
def tcp_connection(port, nums_worker):

    server1socket = socket(AF_INET, SOCK_STREAM)
    server1socket.bind(('', port))
    server1socket.listen(nums_worker)
    print('The Parameter Server Is Ready: ')

    return server1socket


def send_init_parameters(connectionsocket, parameters, worker_id):

    dumps_parameters = pk.dumps(parameters)
    connectionsocket.send(dumps_parameters)
    print("Send the initial parameters to the worker{} success ! ".format(worker_id))


# 创建队列
def create_queue(nums_worker):
    queue_dict = {}
    for i in range(nums_worker + 1):
        queue_dict["queue" + str(i + 1)] = Queue()
    return queue_dict


def create_utils(sockets):
    t = {}
    ep = {}
    for i in sockets:
        t[str(i)] = 0
        ep[str(i)] = 0

    return t, ep


def create_backup(sockets, parameters, grads):

    backup = {}
    meansquare = {}
    for i in sockets:
        backup[str(i)] = parameters
        meansquare[str(i)] = grads
    return backup, meansquare


def create_time_table(sockets):

    A = {}
    r_star = {}
    t0 = 0
    t1 = 0
    for i in sockets:
        A[str(i)] = [t0, t1]
        r_star[str(i)] = 0

    return A, r_star


def create_grads_dict(len_of_parameters):

    grads = {}
    momentum_grads = {}
    for i in range(len_of_parameters):
        grads['dw' + str(i+1)] = 0
        grads['db' + str(i+1)] = 0
        momentum_grads['dw' + str(i+1)] = 0
        momentum_grads['db' + str(i+1)] = 0

    return grads, momentum_grads


def create_placeholder():
    # weights samples
    x = tf.placeholder(tf.float32, [None, 8, 8, 5361], name='input_x')

    return x


def inference(input_tensor):
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
        w54 = tf.get_variable('fcn1-weight', [flatten.shape[1], 102500], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z54 = tf.matmul(flatten, w54)
        y54_hat = tf.reshape(z54, [-1, 102500])

    # 输入切片
    # input_tensor = tf.reduce_mean(input_tensor, axis=0, keep_dims=True)
    input_tensor = tf.reshape(input_tensor, [-1, 343104])

    # init_conv
    eval_w1 = input_tensor[0, 0: 864]
    eval_w1 = tf.reshape(eval_w1, [3, 3, 3, 32])
    eval_b1 = input_tensor[0, 864: 896]

    # unit-1
    # shortcut1
    eval_w2 = input_tensor[0, 896: 4992]
    eval_w2 = tf.reshape(eval_w2, [1, 1, 32, 128])
    eval_b2 = input_tensor[0, 4992: 5120]

    # unit-1-0
    eval_w3 = input_tensor[0, 5120: 6144]
    eval_w3 = tf.reshape(eval_w3, [1, 1, 32, 32])
    eval_b3 = input_tensor[0, 6144: 6176]

    eval_w4 = input_tensor[0, 6176: 15392]
    eval_w4 = tf.reshape(eval_w4, [3, 3, 32, 32])
    eval_b4 = input_tensor[0, 15392: 15424]

    eval_w5 = input_tensor[0, 15424: 19520]
    eval_w5 = tf.reshape(eval_w5, [1, 1, 32, 128])
    eval_b5 = input_tensor[0, 19520: 19648]

    # unit-1-1
    eval_w6 = input_tensor[0, 19648: 23744]
    eval_w6 = tf.reshape(eval_w6, [1, 1, 128, 32])
    eval_b6 = input_tensor[0, 23744: 23776]

    eval_w7 = input_tensor[0, 23776: 32992]
    eval_w7 = tf.reshape(eval_w7, [3, 3, 32, 32])
    eval_b7 = input_tensor[0, 32992: 33024]

    eval_w8 = input_tensor[0, 33024: 37120]
    eval_w8 = tf.reshape(eval_w8, [1, 1, 32, 128])
    eval_b8 = input_tensor[0, 37120: 37248]

    # unit-1-2
    eval_w9 = input_tensor[0, 37248: 41344]
    eval_w9 = tf.reshape(eval_w9, [1, 1, 128, 32])
    eval_b9 = input_tensor[0, 41344: 41376]

    eval_w10 = input_tensor[0, 41376: 50592]
    eval_w10 = tf.reshape(eval_w10, [3, 3, 32, 32])
    eval_b10 = input_tensor[0, 50592: 50624]

    eval_w11 = input_tensor[0, 50624: 54720]
    eval_w11 = tf.reshape(eval_w11, [1, 1, 32, 128])
    eval_b11 = input_tensor[0, 54720: 54848]

    # unit-2
    # shortcut2
    eval_w12 = input_tensor[0, 54848: 87616]
    eval_w12 = tf.reshape(eval_w12, [1, 1, 128, 256])
    eval_b12 = input_tensor[0, 87616: 87872]

    # unit-2-0
    eval_w13 = input_tensor[0, 87872: 96064]
    eval_w13 = tf.reshape(eval_w13, [1, 1, 128, 64])
    eval_b13 = input_tensor[0, 96064: 96128]

    eval_w14 = input_tensor[0, 96128: 132992]
    eval_w14 = tf.reshape(eval_w14, [3, 3, 64, 64])
    eval_b14 = input_tensor[0, 132992: 133056]

    eval_w15 = input_tensor[0, 133056: 149440]
    eval_w15 = tf.reshape(eval_w15, [1, 1, 64, 256])
    eval_b15 = input_tensor[0, 149440: 149696]

    # unit-2-1
    eval_w16 = input_tensor[0, 149696: 166080]
    eval_w16 = tf.reshape(eval_w16, [1, 1, 256, 64])
    eval_b16 = input_tensor[0, 166080: 166144]

    eval_w17 = input_tensor[0, 166144: 203008]
    eval_w17 = tf.reshape(eval_w17, [3, 3, 64, 64])
    eval_b17 = input_tensor[0, 203008: 203072]

    eval_w18 = input_tensor[0, 203072: 219456]
    eval_w18 = tf.reshape(eval_w18, [1, 1, 64, 256])
    eval_b18 = input_tensor[0, 219456: 219712]

    # unit-2-2
    eval_w19 = input_tensor[0, 219712: 236096]
    eval_w19 = tf.reshape(eval_w19, [1, 1, 256, 64])
    eval_b19 = input_tensor[0, 236096: 236160]

    eval_w20 = input_tensor[0, 236160: 273024]
    eval_w20 = tf.reshape(eval_w20, [3, 3, 64, 64])
    eval_b20 = input_tensor[0, 273024: 273088]

    eval_w21 = input_tensor[0, 273088: 289472]
    eval_w21 = tf.reshape(eval_w21, [1, 1, 64, 256])
    eval_b21 = input_tensor[0, 289472: 289728]

    # unit-2-3
    eval_w22 = input_tensor[0, 289728: 306112]
    eval_w22 = tf.reshape(eval_w22, [1, 1, 256, 64])
    eval_b22 = input_tensor[0, 306112: 306176]

    eval_w23 = input_tensor[0, 306176: 343040]
    eval_w23 = tf.reshape(eval_w23, [3, 3, 64, 64])
    eval_b23 = input_tensor[0, 343040: 343104]

    # uint-2-3
    decoder_conv_w24 = y24_hat[0, 0: 16384]
    decoder_conv_w24 = tf.reshape(decoder_conv_w24, [1, 1, 64, 256])
    decoder_conv_b24 = y24_hat[0, 16384: 16640]

    # unit-3
    # shortcut3
    decoder_conv_w25 = y25_hat[0, 0: 131072]
    decoder_conv_w25 = tf.reshape(decoder_conv_w25, [1, 1, 256, 512])
    decoder_conv_b25 = y25_hat[0, 131072: 131584]

    # unit-3-0
    decoder_conv_w26 = y26_hat[0, 0: 32768]
    decoder_conv_w26 = tf.reshape(decoder_conv_w26, [1, 1, 256, 128])
    decoder_conv_b26 = y26_hat[0, 32768: 32896]

    decoder_conv_w27 = y27_hat[0, 0: 147456]
    decoder_conv_w27 = tf.reshape(decoder_conv_w27, [3, 3, 128, 128])
    decoder_conv_b27 = y27_hat[0, 147456: 147584]

    decoder_conv_w28 = y28_hat[0, 0: 65536]
    decoder_conv_w28 = tf.reshape(decoder_conv_w28, [1, 1, 128, 512])
    decoder_conv_b28 = y28_hat[0, 65536: 66048]

    # unit-3-1
    decoder_conv_w29 = y29_hat[0, 0: 65536]
    decoder_conv_w29 = tf.reshape(decoder_conv_w29, [1, 1, 512, 128])
    decoder_conv_b29 = y29_hat[0, 65536: 65664]

    decoder_conv_w30 = y30_hat[0, 0: 147456]
    decoder_conv_w30 = tf.reshape(decoder_conv_w30, [3, 3, 128, 128])
    decoder_conv_b30 = y30_hat[0, 147456: 147584]

    decoder_conv_w31 = y31_hat[0, 0: 65536]
    decoder_conv_w31 = tf.reshape(decoder_conv_w31, [1, 1, 128, 512])
    decoder_conv_b31 = y31_hat[0, 65536: 66048]

    # unit-3-2
    decoder_conv_w32 = y32_hat[0, 0: 65536]
    decoder_conv_w32 = tf.reshape(decoder_conv_w32, [1, 1, 512, 128])
    decoder_conv_b32 = y32_hat[0, 65536:  65664]

    decoder_conv_w33 = y33_hat[0, 0: 147456]
    decoder_conv_w33 = tf.reshape(decoder_conv_w33, [3, 3, 128, 128])
    decoder_conv_b33 = y33_hat[0, 147456: 147584]

    decoder_conv_w34 = y34_hat[0, 0: 65536]
    decoder_conv_w34 = tf.reshape(decoder_conv_w34, [1, 1, 128, 512])
    decoder_conv_b34 = y34_hat[0, 65536: 66048]

    # unit-3-3
    decoder_conv_w35 = y35_hat[0, 0: 65536]
    decoder_conv_w35 = tf.reshape(decoder_conv_w35, [1, 1, 512, 128])
    decoder_conv_b35 = y35_hat[0, 65536:  65664]

    decoder_conv_w36 = y36_hat[0, 0: 147456]
    decoder_conv_w36 = tf.reshape(decoder_conv_w36, [3, 3, 128, 128])
    decoder_conv_b36 = y36_hat[0, 147456: 147584]

    decoder_conv_w37 = y37_hat[0, 0: 65536]
    decoder_conv_w37 = tf.reshape(decoder_conv_w37, [1, 1, 128, 512])
    decoder_conv_b37 = y37_hat[0, 65536: 66048]

    # unit-3-4
    decoder_conv_w38 = y38_hat[0, 0: 65536]
    decoder_conv_w38 = tf.reshape(decoder_conv_w38, [1, 1, 512, 128])
    decoder_conv_b38 = y38_hat[0, 65536:  65664]

    decoder_conv_w39 = y39_hat[0, 0: 147456]
    decoder_conv_w39 = tf.reshape(decoder_conv_w39, [3, 3, 128, 128])
    decoder_conv_b39 = y39_hat[0, 147456: 147584]

    decoder_conv_w40 = y40_hat[0, 0: 65536]
    decoder_conv_w40 = tf.reshape(decoder_conv_w40, [1, 1, 128, 512])
    decoder_conv_b40 = y40_hat[0, 65536: 66048]

    # unit-3-5
    decoder_conv_w41 = y41_hat[0, 0: 65536]
    decoder_conv_w41 = tf.reshape(decoder_conv_w41, [1, 1, 512, 128])
    decoder_conv_b41 = y41_hat[0, 65536:  65664]

    decoder_conv_w42 = y42_hat[0, 0: 147456]
    decoder_conv_w42 = tf.reshape(decoder_conv_w42, [3, 3, 128, 128])
    decoder_conv_b42 = y42_hat[0, 147456: 147584]

    decoder_conv_w43 = y43_hat[0, 0: 65536]
    decoder_conv_w43 = tf.reshape(decoder_conv_w43, [1, 1, 128, 512])
    decoder_conv_b43 = y43_hat[0, 65536: 66048]

    # unit-4
    # shortcut4
    decoder_conv_w44 = y44_hat[0, 0: 524288]
    decoder_conv_w44 = tf.reshape(decoder_conv_w44, [1, 1, 512, 1024])
    decoder_conv_b44 = y44_hat[0, 524288: 525312]

    # unit-4-0
    decoder_conv_w45 = y45_hat[0, 0: 131072]
    decoder_conv_w45 = tf.reshape(decoder_conv_w45, [1, 1, 512, 256])
    decoder_conv_b45 = y45_hat[0, 131072:  131328]

    decoder_conv_w46 = y46_hat[0, 0: 589824]
    decoder_conv_w46 = tf.reshape(decoder_conv_w46, [3, 3, 256, 256])
    decoder_conv_b46 = y46_hat[0, 589824: 590080]

    decoder_conv_w47 = y47_hat[0, 0: 262144]
    decoder_conv_w47 = tf.reshape(decoder_conv_w47, [1, 1, 256, 1024])
    decoder_conv_b47 = y47_hat[0, 262144: 263168]

    # unit-4-1
    decoder_conv_w48 = y48_hat[0, 0: 262144]
    decoder_conv_w48 = tf.reshape(decoder_conv_w48, [1, 1, 1024, 256])
    decoder_conv_b48 = y48_hat[0, 262144:  262400]

    decoder_conv_w49 = y49_hat[0, 0: 589824]
    decoder_conv_w49 = tf.reshape(decoder_conv_w49, [3, 3, 256, 256])
    decoder_conv_b49 = y49_hat[0, 589824: 590080]

    decoder_conv_w50 = y50_hat[0, 0: 262144]
    decoder_conv_w50 = tf.reshape(decoder_conv_w50, [1, 1, 256, 1024])
    decoder_conv_b50 = y50_hat[0, 262144: 263168]

    # unit-4-2
    decoder_conv_w51 = y51_hat[0, 0: 262144]
    decoder_conv_w51 = tf.reshape(decoder_conv_w51, [1, 1, 1024, 256])
    decoder_conv_b51 = y51_hat[0, 262144:  262400]

    decoder_conv_w52 = y52_hat[0, 0: 589824]
    decoder_conv_w52= tf.reshape(decoder_conv_w52, [3, 3, 256, 256])
    decoder_conv_b52 = y52_hat[0, 589824: 590080]

    decoder_conv_w53 = y53_hat[0, 0: 262144]
    decoder_conv_w53 = tf.reshape(decoder_conv_w53, [1, 1, 256, 1024])
    decoder_conv_b53 = y53_hat[0, 262144: 263168]

    decoder_fcn_w54 = y54_hat[0, 0: 102400]
    decoder_fcn_w54 = tf.reshape(decoder_fcn_w54, [1024, 100])
    decoder_fcn_b54 = y54_hat[0, 102400: 102500]

    parameters = {'w1': eval_w1, 'w2': eval_w2, 'w3': eval_w3, 'w4': eval_w4, 'w5': eval_w5, 'w6': eval_w6,
                  'w7': eval_w7, 'w8': eval_w8, 'w9': eval_w9, 'w10': eval_w10, 'w11': eval_w11, 'w12': eval_w12,
                  'w13': eval_w13, 'w14': eval_w14, 'w15': eval_w15, 'w16': eval_w16,
                  'w17': eval_w17, 'w18': eval_w18, 'w19': eval_w19, 'w20': eval_w20,
                  'w21': eval_w21, 'w22': eval_w22, 'w23': eval_w23, 'w24': decoder_conv_w24,
                  'w25': decoder_conv_w25, 'w26': decoder_conv_w26, 'w27': decoder_conv_w27, 'w28': decoder_conv_w28,
                  'w29': decoder_conv_w29, 'w30': decoder_conv_w30, 'w31': decoder_conv_w31, 'w32': decoder_conv_w32,
                  'w33': decoder_conv_w33, 'w34': decoder_conv_w34, 'w35': decoder_conv_w35, 'w36': decoder_conv_w36,
                  'w37': decoder_conv_w37, 'w38': decoder_conv_w38, 'w39': decoder_conv_w39, 'w40': decoder_conv_w40,
                  'w41': decoder_conv_w41, 'w42': decoder_conv_w42, 'w43': decoder_conv_w43, 'w44': decoder_conv_w44,
                  'w45': decoder_conv_w45, 'w46': decoder_conv_w46, 'w47': decoder_conv_w47, 'w48': decoder_conv_w48,
                  'w49': decoder_conv_w49, 'w50': decoder_conv_w50, 'w51': decoder_conv_w51, 'w52': decoder_conv_w52,
                  'w53': decoder_conv_w53, 'w54': decoder_fcn_w54,

                  'b1': eval_b1, 'b2': eval_b2, 'b3': eval_b3, 'b4': eval_b4, 'b5': eval_b5, 'b6': eval_b6,
                  'b7': eval_b7, 'b8': eval_b8, 'b9': eval_b9, 'b10': eval_b10, 'b11': eval_b11, 'b12': eval_b12,
                  'b13': eval_b13, 'b14': eval_b14,'b15': eval_b15, 'b16': eval_b16,
                  'b17': eval_b17, 'b18': eval_b18, 'b19': eval_b19, 'b20': eval_b20,
                  'b21': eval_b21, 'b22': eval_b22, 'b23': eval_b23, 'b24': decoder_conv_b24,
                  'b25': decoder_conv_b25, 'b26': decoder_conv_b26, 'b27': decoder_conv_b27, 'b28': decoder_conv_b28,
                  'b29': decoder_conv_b29, 'b30': decoder_conv_b30, 'b31': decoder_conv_b31, 'b32': decoder_conv_b32,
                  'b33': decoder_conv_b33, 'b34': decoder_conv_b34, 'b35': decoder_conv_b35, 'b36': decoder_conv_b36,
                  'b37': decoder_conv_b37, 'b38': decoder_conv_b38, 'b39': decoder_conv_b39, 'b40': decoder_conv_b40,
                  'b41': decoder_conv_b41, 'b42': decoder_conv_b42, 'b43': decoder_conv_b43, 'b44': decoder_conv_b44,
                  'b45': decoder_conv_b45, 'b46': decoder_conv_b46, 'b47': decoder_conv_b47, 'b48': decoder_conv_b48,
                  'b49': decoder_conv_b49, 'b50': decoder_conv_b50, 'b51': decoder_conv_b51, 'b52': decoder_conv_b52,
                  'b53': decoder_conv_b53, 'b54': decoder_fcn_b54,

                  }

    return parameters