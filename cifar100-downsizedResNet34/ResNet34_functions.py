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
    x = tf.placeholder(tf.float32, [None, 8, 8, 5498], name='input_x')

    return x


def inference(input_tensor):
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
        w17 = tf.get_variable('conv17-weight', [1, 1, 1, 1154], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z17 = tf.nn.conv2d(a4, w17, strides=[1, 1, 1, 1], padding='VALID')
        y17_hat = tf.reshape(z17, [-1, 73856])

        w18 = tf.get_variable('conv18-weight', [1, 1, 1, 1154], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z18 = tf.nn.conv2d(a4, w18, strides=[1, 1, 1, 1], padding='VALID')
        y18_hat = tf.reshape(z18, [-1, 73856])

        w19 = tf.get_variable('conv19-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z19 = tf.nn.conv2d(a4, w19, strides=[1, 1, 1, 1], padding='VALID')
        y19_hat = tf.reshape(z19, [-1, 147584])

        w20 = tf.get_variable('conv20-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z20 = tf.nn.conv2d(a4, w20, strides=[1, 1, 1, 1], padding='VALID')
        y20_hat = tf.reshape(z20, [-1, 147584])

        w21 = tf.get_variable('conv21-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z21 = tf.nn.conv2d(a4, w21, strides=[1, 1, 1, 1], padding='VALID')
        y21_hat = tf.reshape(z21, [-1, 147584])

        w22 = tf.get_variable('conv22-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z22 = tf.nn.conv2d(a4, w22, strides=[1, 1, 1, 1], padding='VALID')
        y22_hat = tf.reshape(z22, [-1, 147584])

        w23 = tf.get_variable('conv23-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z23 = tf.nn.conv2d(a4, w23, strides=[1, 1, 1, 1], padding='VALID')
        y23_hat = tf.reshape(z23, [-1, 147584])

        w24 = tf.get_variable('conv24-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z24 = tf.nn.conv2d(a4, w24, strides=[1, 1, 1, 1], padding='VALID')
        y24_hat = tf.reshape(z24, [-1, 147584])

        w25 = tf.get_variable('conv25-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z25 = tf.nn.conv2d(a4, w25, strides=[1, 1, 1, 1], padding='VALID')
        y25_hat = tf.reshape(z25, [-1, 147584])

        w26 = tf.get_variable('conv26-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z26 = tf.nn.conv2d(a4, w26, strides=[1, 1, 1, 1], padding='VALID')
        y26_hat = tf.reshape(z26, [-1, 147584])

        w27 = tf.get_variable('conv27-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z27 = tf.nn.conv2d(a4, w27, strides=[1, 1, 1, 1], padding='VALID')
        y27_hat = tf.reshape(z27, [-1, 147584])

        w28 = tf.get_variable('conv28-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z28 = tf.nn.conv2d(a4, w28, strides=[1, 1, 1, 1], padding='VALID')
        y28_hat = tf.reshape(z28, [-1, 147584])

        w29 = tf.get_variable('conv29-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z29 = tf.nn.conv2d(a4, w29, strides=[1, 1, 1, 1], padding='VALID')
        y29_hat = tf.reshape(z29, [-1, 147584])

        # unit-4
        w30 = tf.get_variable('conv30-weight', [1, 1, 1, 4612], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z30 = tf.nn.conv2d(a4, w30, strides=[1, 1, 1, 1], padding='VALID')
        y30_hat = tf.reshape(z30, [-1, 295168])

        w31 = tf.get_variable('conv31-weight', [1, 1, 1, 4612], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z31 = tf.nn.conv2d(a4, w31, strides=[1, 1, 1, 1], padding='VALID')
        y31_hat = tf.reshape(z31, [-1, 295168])

        w32 = tf.get_variable('conv32-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z32 = tf.nn.conv2d(a4, w32, strides=[1, 1, 1, 1], padding='VALID')
        y32_hat = tf.reshape(z32, [-1, 590080])

        w33 = tf.get_variable('conv33-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z33 = tf.nn.conv2d(a4, w33, strides=[1, 1, 1, 1], padding='VALID')
        y33_hat = tf.reshape(z33, [-1, 590080])

        w34 = tf.get_variable('conv34-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z34 = tf.nn.conv2d(a4, w34, strides=[1, 1, 1, 1], padding='VALID')
        y34_hat = tf.reshape(z34, [-1, 590080])

        w35 = tf.get_variable('conv35-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z35 = tf.nn.conv2d(a4, w35, strides=[1, 1, 1, 1], padding='VALID')
        y35_hat = tf.reshape(z35, [-1, 590080])

        w36 = tf.get_variable('conv36-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
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
    eval_w1 = input_tensor[0, 0: 864]
    eval_w1 = tf.reshape(eval_w1, [3, 3, 3, 32])
    eval_b1 = input_tensor[0, 864: 896]

    # unit-1
    eval_w2 = input_tensor[0, 896: 10112]
    eval_w2 = tf.reshape(eval_w2, [3, 3, 32, 32])
    eval_b2 = input_tensor[0, 10112: 10144]

    eval_w3 = input_tensor[0, 10144: 19360]
    eval_w3 = tf.reshape(eval_w3, [3, 3, 32, 32])
    eval_b3 = input_tensor[0, 19360: 19392]

    eval_w4 = input_tensor[0, 19392: 28608]
    eval_w4 = tf.reshape(eval_w4, [3, 3, 32, 32])
    eval_b4 = input_tensor[0, 28608: 28640]

    eval_w5 = input_tensor[0, 28640: 37856]
    eval_w5 = tf.reshape(eval_w5, [3, 3, 32, 32])
    eval_b5 = input_tensor[0, 37856: 37888]

    eval_w6 = input_tensor[0, 37888: 47104]
    eval_w6 = tf.reshape(eval_w6, [3, 3, 32, 32])
    eval_b6 = input_tensor[0, 47104: 47136]

    eval_w7 = input_tensor[0, 47136: 56352]
    eval_w7 = tf.reshape(eval_w7, [3, 3, 32, 32])
    eval_b7 = input_tensor[0, 56352: 56384]

    # unit-2
    eval_w8 = input_tensor[0, 56384: 74816]
    eval_w8 = tf.reshape(eval_w8, [3, 3, 32, 64])
    eval_b8 = input_tensor[0, 74816: 74880]

    eval_w9 = input_tensor[0, 74880: 93312]
    eval_w9 = tf.reshape(eval_w9, [3, 3, 32, 64])
    eval_b9 = input_tensor[0, 93312: 93376]

    eval_w10 = input_tensor[0, 93376: 130240]
    eval_w10 = tf.reshape(eval_w10, [3, 3, 64, 64])
    eval_b10 = input_tensor[0, 130240: 130304]

    eval_w11 = input_tensor[0, 130304: 167168]
    eval_w11 = tf.reshape(eval_w11, [3, 3, 64, 64])
    eval_b11 = input_tensor[0, 167168: 167232]

    eval_w12 = input_tensor[0, 167232: 204096]
    eval_w12 = tf.reshape(eval_w12, [3, 3, 64, 64])
    eval_b12 = input_tensor[0, 204096: 204160]

    eval_w13 = input_tensor[0, 204160: 241024]
    eval_w13 = tf.reshape(eval_w13, [3, 3, 64, 64])
    eval_b13 = input_tensor[0, 241024: 241088]

    eval_w14 = input_tensor[0, 241088: 277952]
    eval_w14 = tf.reshape(eval_w14, [3, 3, 64, 64])
    eval_b14 = input_tensor[0, 277952: 278016]

    eval_w15 = input_tensor[0, 278016: 314880]
    eval_w15 = tf.reshape(eval_w15, [3, 3, 64, 64])
    eval_b15 = input_tensor[0, 314880: 314944]

    eval_w16 = input_tensor[0, 314944: 351808]
    eval_w16 = tf.reshape(eval_w16, [3, 3, 64, 64])
    eval_b16 = input_tensor[0, 351808: 351872]


    # unit-3
    decoder_conv_w17 = y17_hat[0, 0: 73728]
    decoder_conv_w17 = tf.reshape(decoder_conv_w17, [3, 3, 64, 128])
    decoder_conv_b17 = y17_hat[0, 73728: 73856]

    decoder_conv_w18 = y18_hat[0, 0: 73728]
    decoder_conv_w18 = tf.reshape(decoder_conv_w18, [3, 3, 64, 128])
    decoder_conv_b18 = y18_hat[0, 73728: 73856]

    decoder_conv_w19 = y19_hat[0, 0: 147456]
    decoder_conv_w19 = tf.reshape(decoder_conv_w19, [3, 3, 128, 128])
    decoder_conv_b19 = y19_hat[0, 147456: 147584]

    decoder_conv_w20 = y20_hat[0, 0: 147456]
    decoder_conv_w20 = tf.reshape(decoder_conv_w20, [3, 3, 128, 128])
    decoder_conv_b20 = y20_hat[0, 147456: 147584]

    decoder_conv_w21 = y21_hat[0, 0: 147456]
    decoder_conv_w21 = tf.reshape(decoder_conv_w21, [3, 3, 128, 128])
    decoder_conv_b21 = y21_hat[0, 147456: 147584]

    decoder_conv_w22 = y22_hat[0, 0: 147456]
    decoder_conv_w22 = tf.reshape(decoder_conv_w22, [3, 3, 128, 128])
    decoder_conv_b22 = y22_hat[0, 147456: 147584]

    decoder_conv_w23 = y23_hat[0, 0: 147456]
    decoder_conv_w23 = tf.reshape(decoder_conv_w23, [3, 3, 128, 128])
    decoder_conv_b23 = y23_hat[0, 147456: 147584]

    decoder_conv_w24 = y24_hat[0, 0: 147456]
    decoder_conv_w24 = tf.reshape(decoder_conv_w24, [3, 3, 128, 128])
    decoder_conv_b24 = y24_hat[0, 147456: 147584]

    decoder_conv_w25 = y25_hat[0, 0: 147456]
    decoder_conv_w25 = tf.reshape(decoder_conv_w25, [3, 3, 128, 128])
    decoder_conv_b25 = y25_hat[0, 147456: 147584]

    decoder_conv_w26 = y26_hat[0, 0: 147456]
    decoder_conv_w26 = tf.reshape(decoder_conv_w26, [3, 3, 128, 128])
    decoder_conv_b26 = y26_hat[0, 147456: 147584]

    decoder_conv_w27 = y27_hat[0, 0: 147456]
    decoder_conv_w27 = tf.reshape(decoder_conv_w27, [3, 3, 128, 128])
    decoder_conv_b27 = y27_hat[0, 147456: 147584]

    decoder_conv_w28 = y28_hat[0, 0: 147456]
    decoder_conv_w28 = tf.reshape(decoder_conv_w28, [3, 3, 128, 128])
    decoder_conv_b28 = y28_hat[0, 147456: 147584]

    decoder_conv_w29 = y29_hat[0, 0: 147456]
    decoder_conv_w29 = tf.reshape(decoder_conv_w29, [3, 3, 128, 128])
    decoder_conv_b29 = y29_hat[0, 147456: 147584]

    # uint-4
    decoder_conv_w30 = y30_hat[0, 0: 294912]
    decoder_conv_w30 = tf.reshape(decoder_conv_w30, [3, 3, 128, 256])
    decoder_conv_b30 = y30_hat[0, 294912: 295168]

    decoder_conv_w31 = y31_hat[0, 0: 294912]
    decoder_conv_w31 = tf.reshape(decoder_conv_w31, [3, 3, 128, 256])
    decoder_conv_b31 = y31_hat[0, 294912: 295168]

    decoder_conv_w32 = y32_hat[0, 0: 589824]
    decoder_conv_w32 = tf.reshape(decoder_conv_w32, [3, 3, 256, 256])
    decoder_conv_b32 = y32_hat[0, 589824: 590080]

    decoder_conv_w33 = y33_hat[0, 0: 589824]
    decoder_conv_w33 = tf.reshape(decoder_conv_w33, [3, 3, 256, 256])
    decoder_conv_b33 = y33_hat[0, 589824: 590080]

    decoder_conv_w34 = y34_hat[0, 0: 589824]
    decoder_conv_w34 = tf.reshape(decoder_conv_w34, [3, 3, 256, 256])
    decoder_conv_b34 = y34_hat[0, 589824: 590080]

    decoder_conv_w35 = y35_hat[0, 0: 589824]
    decoder_conv_w35 = tf.reshape(decoder_conv_w35, [3, 3, 256, 256])
    decoder_conv_b35 = y35_hat[0, 589824: 590080]

    decoder_conv_w36 = y36_hat[0, 0: 589824]
    decoder_conv_w36 = tf.reshape(decoder_conv_w36, [3, 3, 256, 256])
    decoder_conv_b36 = y36_hat[0, 589824: 590080]

    decoder_fcn_w37 = y37_hat[0, 0: 25600]
    decoder_fcn_w37 = tf.reshape(decoder_fcn_w37, [256, 100])
    decoder_fcn_b37 = y37_hat[0, 25600: 25700]

    parameters = {'w1': eval_w1, 'w2': eval_w2, 'w3': eval_w3, 'w4': eval_w4, 'w5': eval_w5, 'w6': eval_w6,
                  'w7': eval_w7, 'w8': eval_w8, 'w9': eval_w9, 'w10': eval_w10, 'w11': eval_w11, 'w12': eval_w12,
                  'w13': eval_w13, 'w14': eval_w14, 'w15': eval_w15, 'w16': eval_w16,
                  'w17': decoder_conv_w17, 'w18': decoder_conv_w18, 'w19': decoder_conv_w19, 'w20': decoder_conv_w20,
                  'w21': decoder_conv_w21, 'w22': decoder_conv_w22, 'w23': decoder_conv_w23, 'w24': decoder_conv_w24,
                  'w25': decoder_conv_w25, 'w26': decoder_conv_w26, 'w27': decoder_conv_w27, 'w28': decoder_conv_w28,
                  'w29': decoder_conv_w29, 'w30': decoder_conv_w30, 'w31': decoder_conv_w31, 'w32': decoder_conv_w32,
                  'w33': decoder_conv_w33, 'w34': decoder_conv_w34, 'w35': decoder_conv_w35, 'w36': decoder_conv_w36,
                  'w37': decoder_fcn_w37,

                  'b1': eval_b1, 'b2': eval_b2, 'b3': eval_b3, 'b4': eval_b4, 'b5': eval_b5, 'b6': eval_b6,
                  'b7': eval_b7, 'b8': eval_b8, 'b9': eval_b9, 'b10': eval_b10, 'b11': eval_b11, 'b12': eval_b12,
                  'b13': eval_b13, 'b14': eval_b14 ,'b15': eval_b15, 'b16': eval_b16,
                  'b17': decoder_conv_b17, 'b18': decoder_conv_b18, 'b19': decoder_conv_b19, 'b20': decoder_conv_b20,
                  'b21': decoder_conv_b21, 'b22': decoder_conv_b22, 'b23': decoder_conv_b23, 'b24': decoder_conv_b24,
                  'b25': decoder_conv_b25, 'b26': decoder_conv_b26, 'b27': decoder_conv_b27, 'b28': decoder_conv_b28,
                  'b29': decoder_conv_b29, 'b30': decoder_conv_b30, 'b31': decoder_conv_b31, 'b32': decoder_conv_b32,
                  'b33': decoder_conv_b33, 'b34': decoder_conv_b34, 'b35': decoder_conv_b35, 'b36': decoder_conv_b36,
                  'b37': decoder_fcn_b37,
                  }

    return parameters