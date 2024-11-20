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
    x = tf.placeholder(tf.float32, [None, 8, 8, 4065], name='input_x')

    return x


def inference(input_tensor):
    # 1-prediction network

    is_train = tf.placeholder_with_default(False, (), 'is_train')
    # 1-prediction network
    # Layer-1: Conv-(3, 3, 1188, 6)
    with tf.variable_scope('layer1-Conv1'):
        w1 = tf.get_variable('weight', [3, 3, 4065, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
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
        a3 = tf.nn.relu(z3)
        a3 = SE_block(a3, ratio=4)

    # Pool layer
    with tf.variable_scope('pool3-Layer'):
        pool3 = tf.nn.max_pool(a3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    flatten = tf.contrib.layers.flatten(pool3)
    print(flatten.shape)

    # Decoder-1: decode vgg11 convolution layer (L3-L8)
    with tf.variable_scope('Decoder-1'):
        w4 = tf.get_variable('decode-weight-1', [flatten.shape[1], 64],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        b4 = tf.get_variable('decode-bias-1', [64], initializer=tf.constant_initializer(0.1))
        z4 = tf.matmul(flatten, w4) + b4
        a4 = tf.nn.relu(z4)
        # a4_cp = a4
        a4 = tf.reshape(a4, [-1, 8, 8, 1])

        # decode1_w5 = tf.get_variable('decode-weight-2', [a4_cp.shape[1], 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # decode1_b5 = tf.get_variable('decode-bias-2', [64], initializer=tf.constant_initializer(0.1))
        # decode1_z5 = tf.matmul(a4_cp, decode1_w5) + decode1_b5
        # decode1_a5 = tf.nn.relu(decode1_z5)
        # decode1_a5 = tf.reshape(decode1_a5, [-1, 8, 8, 1])

        # max-pool1

        # decode_w3 = tf.get_variable('decode-conv3-weight', [1, 1, 1, 73856], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # decode_z3 = tf.nn.conv2d(a4, decode_w3, strides=[1, 1, 1, 1], padding='VALID')
        # y3_hat = tf.reshape(decode_z3, [-1, 73856])

        # decode_w4 = tf.get_variable('decode-conv4-weight', [1, 1, 1, 147584], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # decode_z4 = tf.nn.conv2d(a4, decode_w4, strides=[1, 1, 1, 1], padding='VALID')
        # y4_hat = tf.reshape(decode_z4, [-1, 147584])

        # max-pool2

        w5 = tf.get_variable('conv5-weight', [1, 1, 1, 4612], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z5 = tf.nn.conv2d(a4, w5, strides=[1, 1, 1, 1], padding='VALID')
        y5_hat = tf.reshape(z5, [-1, 295168])

        w6 = tf.get_variable('conv6-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z6 = tf.nn.conv2d(a4, w6, strides=[1, 1, 1, 1], padding='VALID')
        y6_hat = tf.reshape(z6, [-1, 590080])

        w7 = tf.get_variable('conv7-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z7 = tf.nn.conv2d(a4, w7, strides=[1, 1, 1, 1], padding='VALID')
        y7_hat = tf.reshape(z7, [-1, 590080])

        # max-pool3

        w8 = tf.get_variable('conv8-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z8 = tf.nn.conv2d(a4, w8, strides=[1, 1, 1, 1], padding='VALID')
        y8_hat = tf.reshape(z8, [-1, 590080])

        w9 = tf.get_variable('conv9-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z9 = tf.nn.conv2d(a4, w9, strides=[1, 1, 1, 1], padding='VALID')
        y9_hat = tf.reshape(z9, [-1, 590080])

        # w10 = tf.get_variable('conv10-weight', [1, 1, 1,9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # z10 = tf.nn.conv2d(a4, w10, strides=[1, 1, 1, 1], padding='VALID')
        # y10_hat = tf.reshape(z10, [-1, 590080])

        # max-pool4

        w11 = tf.get_variable('conv11-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z11 = tf.nn.conv2d(a4, w11, strides=[1, 1, 1, 1], padding='VALID')
        y11_hat = tf.reshape(z11, [-1, 590080])

        w12 = tf.get_variable('conv12-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z12 = tf.nn.conv2d(a4, w12, strides=[1, 1, 1, 1], padding='VALID')
        y12_hat = tf.reshape(z12, [-1, 590080])

        w13 = tf.get_variable('conv13-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z13 = tf.nn.conv2d(a4, w13, strides=[1, 1, 1, 1], padding='VALID')
        y13_hat = tf.reshape(z13, [-1, 590080])

        # max-pool5

    # Decoder-2: decode vgg11 fcn layer
    with tf.variable_scope('Decoder-2'):
        decode_fc_w = tf.get_variable('fcn-weight-1', [flatten.shape[1], 64],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        decode_fc_b = tf.get_variable('fcn-biases-1', [64], initializer=tf.constant_initializer(0.1))
        decode_fc_z = tf.matmul(flatten, decode_fc_w) + decode_fc_b
        decode_fc_z = tf.nn.relu(decode_fc_z)

        w14 = tf.get_variable('fcn1-weight', [decode_fc_z.shape[1], 25700],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        z14 = tf.matmul(decode_fc_z, w14)
        y14_hat = tf.reshape(z14, [-1, 25700])

        w15 = tf.get_variable('fcn2-weight', [decode_fc_z.shape[1], 10100],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        z15 = tf.matmul(decode_fc_z, w15)
        y15_hat = tf.reshape(z15, [-1, 10100])

        w16 = tf.get_variable('fcn3-weight', [decode_fc_z.shape[1], 10100],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        z16 = tf.matmul(decode_fc_z, w16)
        y16_hat = tf.reshape(z16, [-1, 10100])

    # 输入切片
    # input_tensor = tf.reduce_mean(input_tensor, axis=0, keep_dims=True)
    input_tensor = tf.reshape(input_tensor, [-1, 260160])

    eval_w1 = input_tensor[0, 0: 1728]
    eval_w1 = tf.reshape(eval_w1, [3, 3, 3, 64])
    eval_b1 = input_tensor[0, 1728: 1792]

    eval_w2 = input_tensor[0, 1792: 38656]
    eval_w2 = tf.reshape(eval_w2, [3, 3, 64, 64])
    eval_b2 = input_tensor[0, 38656: 38720]

    eval_w3 = input_tensor[0, 38720: 112448]
    eval_w3 = tf.reshape(eval_w3, [3, 3, 64, 128])
    eval_b3 = input_tensor[0, 112448: 112576]

    eval_w4 = input_tensor[0, 112576: 260032]
    eval_w4 = tf.reshape(eval_w4, [3, 3, 128, 128])
    eval_b4 = input_tensor[0, 260032: 260160]

    #    eval_w5 = input_tensor[seed, 260160: 555072]
    #    eval_w5 = tf.reshape(eval_w5, [3, 3, 128, 256])
    #    eval_b5 = input_tensor[seed, 555072: 555328]

    #    eval_w6 = input_tensor[seed, 555328: 1145152]
    #    eval_w6 = tf.reshape(eval_w6, [3, 3, 256, 256])
    #    eval_b6 = input_tensor[seed, 1145152: 1145408]

    #    eval_w7 = input_tensor[seed, 1145408: 1735232]
    #    eval_w7 = tf.reshape(eval_w7, [3, 3, 256, 256])
    #    eval_b7 = input_tensor[seed, 1735232: 1735488]

    # Decoder 切片
    # 张量切片: biases = z5[1, 30720:30840]

    # decoder_conv_w3 = y3_hat[seed, 0: 73728]
    # decoder_conv_w3 = tf.reshape(decoder_conv_w3, [3, 3, 64, 128])
    # decoder_conv_b3 = y3_hat[seed, 73728: 73856]

    # decoder_conv_w4 = y4_hat[seed, 0: 147456]
    # decoder_conv_w4 = tf.reshape(decoder_conv_w4, [3, 3, 128, 128])
    # decoder_conv_b4 = y4_hat[seed, 147456: 147584]

    decoder_conv_w5 = y5_hat[0, 0: 294912]
    decoder_conv_w5 = tf.reshape(decoder_conv_w5, [3, 3, 128, 256])
    decoder_conv_b5 = y5_hat[0, 294912: 295168]

    decoder_conv_w6 = y6_hat[0, 0: 589824]
    decoder_conv_w6 = tf.reshape(decoder_conv_w6, [3, 3, 256, 256])
    decoder_conv_b6 = y6_hat[0, 589824: 590080]

    decoder_conv_w7 = y7_hat[0, 0: 589824]
    decoder_conv_w7 = tf.reshape(decoder_conv_w7, [3, 3, 256, 256])
    decoder_conv_b7 = y7_hat[0, 589824: 590080]

    decoder_conv_w8 = y8_hat[0, 0: 589824]
    decoder_conv_w8 = tf.reshape(decoder_conv_w8, [3, 3, 256, 256])
    decoder_conv_b8 = y8_hat[0, 589824: 590080]

    decoder_conv_w9 = y9_hat[0, 0: 589824]
    decoder_conv_w9 = tf.reshape(decoder_conv_w9, [3, 3, 256, 256])
    decoder_conv_b9 = y9_hat[0, 589824: 590080]

    # decoder_conv_w10 = y10_hat[seed, 0: 589824]
    # decoder_conv_w10 = tf.reshape(decoder_conv_w10, [3, 3, 256, 256])
    # decoder_conv_b10 = y10_hat[seed, 589824: 590080]

    decoder_conv_w11 = y11_hat[0, 0: 589824]
    decoder_conv_w11 = tf.reshape(decoder_conv_w11, [3, 3, 256, 256])
    decoder_conv_b11 = y11_hat[0, 589824: 590080]

    decoder_conv_w12 = y12_hat[0, 0: 589824]
    decoder_conv_w12 = tf.reshape(decoder_conv_w12, [3, 3, 256, 256])
    decoder_conv_b12 = y12_hat[0, 589824: 590080]

    decoder_conv_w13 = y13_hat[0, 0: 589824]
    decoder_conv_w13 = tf.reshape(decoder_conv_w13, [3, 3, 256, 256])
    decoder_conv_b13 = y13_hat[0, 589824: 590080]

    decoder_fcn_w1 = y14_hat[0, 0: 25600]
    decoder_fcn_w1 = tf.reshape(decoder_fcn_w1, [256, 100])
    decoder_fcn_b1 = y14_hat[0, 25600: 26700]

    decoder_fcn_w2 = y15_hat[0, 0: 10000]
    decoder_fcn_w2 = tf.reshape(decoder_fcn_w2, [100, 100])
    decoder_fcn_b2 = y15_hat[0, 10000: 10100]

    decoder_fcn_w3 = y16_hat[0, 0: 10000]
    decoder_fcn_w3 = tf.reshape(decoder_fcn_w3, [100, 100])
    decoder_fcn_b3 = y16_hat[0, 10000: 10100]

    parameters = {'w1': eval_w1, 'w2': eval_w2, 'w3': eval_w3, 'w4': eval_w4,
                  'w5': decoder_conv_w5, 'w6': decoder_conv_w6, 'w7': decoder_conv_w7, 'w8': decoder_conv_w8,
                  'w9': decoder_conv_w9, 'w10': decoder_conv_w11, 'w11': decoder_conv_w12, 'w12': decoder_conv_w13,
                  'w13': decoder_fcn_w1, 'w14': decoder_fcn_w2, 'w15': decoder_fcn_w3,
                  'b1': eval_b1, 'b2': eval_b2, 'b3': eval_b3, 'b4': eval_b4,
                  'b5': decoder_conv_b5, 'b6': decoder_conv_b6, 'b7': decoder_conv_b7, 'b8': decoder_conv_b8,
                  'b9': decoder_conv_b9, 'b10': decoder_conv_b11, 'b11': decoder_conv_b12, 'b12': decoder_conv_b13,
                  'b13': decoder_fcn_b1, 'b14': decoder_fcn_b2, 'b15': decoder_fcn_b3}

    return parameters