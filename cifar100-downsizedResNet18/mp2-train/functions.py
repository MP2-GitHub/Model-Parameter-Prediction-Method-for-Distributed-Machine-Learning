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
    x = tf.placeholder(tf.float32, [None, 8, 8, 2901], name='input_x')

    return x


def inference(input_tensor):
    # 1-prediction network
    with tf.variable_scope('layer1-Conv1'):
        w1 = tf.get_variable('weight', [3, 3, 2901, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('biases', [8], initializer=tf.constant_initializer(0.1))
        z1 = tf.nn.bias_add(tf.nn.conv2d(input_tensor, w1, strides=[1, 1, 1, 1], padding='SAME'), b1)
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

    # Decoder-1:
    with tf.variable_scope('Decoder-1'):
        w4 = tf.get_variable('fcn-weight-1', [flatten.shape[1], 64],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        b4 = tf.get_variable('fcn-biases-1', [64], initializer=tf.constant_initializer(0.1))
        z4 = tf.matmul(flatten, w4) + b4
        # z4 = tf.layers.batch_normalization(z4, training=is_train2)
        a4 = tf.nn.relu(z4)
        a4 = tf.reshape(a4, [-1, 8, 8, 1])

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
        w21 = tf.get_variable('fcn1-weight', [flatten.shape[1], 25700],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        z21 = tf.matmul(flatten, w21)
        y21_hat = tf.reshape(z21, [-1, 25700])

    # 输入切片
    # input_tensor = tf.reduce_mean(input_tensor, axis=0, keep_dims=True)
    input_tensor = tf.reshape(input_tensor, [-1, 185664])

    # uint-1
    eval_w1 = input_tensor[0, 0: 864]
    eval_w1 = tf.reshape(eval_w1, [3, 3, 3, 32])
    eval_b1 = input_tensor[0, 864: 896]

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

    # uint-2
    eval_w6 = input_tensor[0, 37888: 56320]
    eval_w6 = tf.reshape(eval_w6, [3, 3, 32, 64])
    eval_b6 = input_tensor[0, 56320: 56384]

    eval_w7 = input_tensor[0, 56384: 74816]
    eval_w7 = tf.reshape(eval_w7, [3, 3, 32, 64])
    eval_b7 = input_tensor[0, 74816: 74880]

    eval_w8 = input_tensor[0, 74880: 111744]
    eval_w8 = tf.reshape(eval_w8, [3, 3, 64, 64])
    eval_b8 = input_tensor[0, 111744: 111808]

    eval_w9 = input_tensor[0, 111808: 148672]
    eval_w9 = tf.reshape(eval_w9, [3, 3, 64, 64])
    eval_b9 = input_tensor[0, 148672: 148736]

    eval_w10 = input_tensor[0, 148736: 185600]
    eval_w10 = tf.reshape(eval_w10, [3, 3, 64, 64])
    eval_b10 = input_tensor[0, 185600: 185664]

    # uint-3
    decoder_conv_w11 = y11_hat[0, 0: 73728]
    decoder_conv_w11 = tf.reshape(decoder_conv_w11, [3, 3, 64, 128])
    decoder_conv_b11 = y11_hat[0, 73728: 73856]

    decoder_conv_w12 = y12_hat[0, 0: 73728]
    decoder_conv_w12 = tf.reshape(decoder_conv_w12, [3, 3, 64, 128])
    decoder_conv_b12 = y12_hat[0, 73728: 73856]

    decoder_conv_w13 = y13_hat[0, 0: 147456]
    decoder_conv_w13 = tf.reshape(decoder_conv_w13, [3, 3, 128, 128])
    decoder_conv_b13 = y13_hat[0, 147456: 147584]

    decoder_conv_w14 = y14_hat[0, 0: 147456]
    decoder_conv_w14 = tf.reshape(decoder_conv_w14, [3, 3, 128, 128])
    decoder_conv_b14 = y14_hat[0, 147456: 147584]

    decoder_conv_w15 = y15_hat[0, 0: 147456]
    decoder_conv_w15 = tf.reshape(decoder_conv_w15, [3, 3, 128, 128])
    decoder_conv_b15 = y15_hat[0, 147456: 147584]

    # uint-4
    decoder_conv_w16 = y16_hat[0, 0: 294912]
    decoder_conv_w16 = tf.reshape(decoder_conv_w16, [3, 3, 128, 256])
    decoder_conv_b16 = y16_hat[0, 294912: 295168]

    decoder_conv_w17 = y17_hat[0, 0: 294912]
    decoder_conv_w17 = tf.reshape(decoder_conv_w17, [3, 3, 128, 256])
    decoder_conv_b17 = y17_hat[0, 294912: 295168]

    decoder_conv_w18 = y18_hat[0, 0: 589824]
    decoder_conv_w18 = tf.reshape(decoder_conv_w18, [3, 3, 256, 256])
    decoder_conv_b18 = y18_hat[0, 589824: 590080]

    decoder_conv_w19 = y19_hat[0, 0: 589824]
    decoder_conv_w19 = tf.reshape(decoder_conv_w19, [3, 3, 256, 256])
    decoder_conv_b19 = y19_hat[0, 589824: 590080]

    decoder_conv_w20 = y20_hat[0, 0: 589824]
    decoder_conv_w20 = tf.reshape(decoder_conv_w20, [3, 3, 256, 256])
    decoder_conv_b20 = y20_hat[0, 589824: 590080]

    decoder_fcn_w21 = y21_hat[0, 0: 25600]
    decoder_fcn_w21 = tf.reshape(decoder_fcn_w21, [256, 100])
    decoder_fcn_b21 = y21_hat[0, 25600: 25700]

    parameters = {'w1': eval_w1, 'w2': eval_w2, 'w3': eval_w3, 'w4': eval_w4, 'w5': eval_w5, 'w6': eval_w6,
                  'w7': eval_w7, 'w8': eval_w8, 'w9': eval_w9, 'w10': eval_w10,
                  'w11': decoder_conv_w11, 'w12': decoder_conv_w12, 'w13': decoder_conv_w13, 'w14': decoder_conv_w14,
                  'w15': decoder_conv_w15, 'w16': decoder_conv_w16, 'w17': decoder_conv_w17, 'w18': decoder_conv_w18,
                  'w19': decoder_conv_w19, 'w20': decoder_conv_w20, 'w21': decoder_fcn_w21,
                  'b1': eval_b1, 'b2': eval_b2, 'b3': eval_b3, 'b4': eval_b4, 'b5': eval_b5, 'b6': eval_b6,
                  'b7': eval_b7, 'b8': eval_b8, 'b9': eval_b9, 'b10': eval_b10,
                  'b11': decoder_conv_b11, 'b12': decoder_conv_b12, 'b13': decoder_conv_b13, 'b14': decoder_conv_b14,
                  'b15': decoder_conv_b15, 'b16': decoder_conv_b16, 'b17': decoder_conv_b17, 'b18': decoder_conv_b18,
                  'b19': decoder_conv_b19, 'b20': decoder_conv_b20, 'b21': decoder_fcn_b21,
                  }

    return parameters