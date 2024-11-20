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
    # print(flatten.shape)
    # with tf.variable_scope('FCN-1'):
    #     w4_1 = tf.get_variable('FCN1-weight', [flatten.shape[1], 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     b4_1 = tf.get_variable('FCN1-biases', [32], initializer=tf.constant_initializer(0.1))
    #     z4_1 = tf.matmul(flatten, w4_1) + b4_1
    #     a4_1 = tf.nn.relu(z4_1)
    #
    # with tf.variable_scope('FCN-2'):
    #     w4_2 = tf.get_variable('FCN2-weight', [32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     b4_2 = tf.get_variable('FCN2-biases', [64], initializer=tf.constant_initializer(0.1))
    #     z4_2 = tf.matmul(a4_1, w4_2) + b4_2
    #     a4_2 = tf.nn.relu(z4_2)
    #
    # with tf.variable_scope('FCN-3'):
    #     w4_3 = tf.get_variable('FCN3-weight', [64, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     b4_3 = tf.get_variable('FCN3-biases', [64], initializer=tf.constant_initializer(0.1))
    #     z4_3 = tf.matmul(a4_2, w4_3) + b4_3
    #     a4_3 = tf.nn.relu(z4_3)

    # Decoder-1: decode vgg11 convolution layer (L3-L8)
    with tf.variable_scope('Decoder-1'):
        w4 = tf.get_variable('fcn-weight-1', [flatten.shape[1], 64],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        b4 = tf.get_variable('fcn-biases-1', [64], initializer=tf.constant_initializer(0.1))
        z4 = tf.matmul(flatten, w4) + b4
        # z4 = tf.layers.batch_normalization(z4, training=is_train2)
        a4 = tf.nn.relu(z4)
        a4 = tf.reshape(a4, [-1, 8, 8, 1])

        # w5 = tf.get_variable('conv1-weight', [1, 1, 1, 1154], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # z5 = tf.nn.conv2d(a4, w5, strides=[1, 1, 1, 1], padding='VALID')
        # # z5 = tf.layers.batch_normalization(z5, training=is_train2)
        # # z5 = tf.reduce_mean(z5, axis=0, keep_dims=True)
        # # z5 = tf.nn.tanh(z5)
        # y1_hat = tf.reshape(z5, [-1, 73856])

        # w6 = tf.get_variable('conv2-weight', [1, 1, 1, 2306], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # z6 = tf.nn.conv2d(a4, w6, strides=[1, 1, 1, 1], padding='VALID')
        # # z6 = tf.layers.batch_normalization(z6, training=is_train2)
        # # z6 = tf.reduce_mean(z6, axis=0, keep_dims=True)
        # # z6 = tf.nn.tanh(z6)
        # y2_hat = tf.reshape(z6, [-1, 147584])

        w7 = tf.get_variable('conv3-weight', [1, 1, 1, 4612], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z7 = tf.nn.conv2d(a4, w7, strides=[1, 1, 1, 1], padding='VALID')
        # z7 = tf.layers.batch_normalization(z7, training=is_train2)
        # z7 = tf.reduce_mean(z7, axis=0, keep_dims=True)
        # z7 = tf.nn.tanh(z7)
        y3_hat = tf.reshape(z7, [-1, 295168])

        w8 = tf.get_variable('conv4-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z8 = tf.nn.conv2d(a4, w8, strides=[1, 1, 1, 1], padding='VALID')
        # z8 = tf.layers.batch_normalization(z8, training=is_train2)
        # z8 = tf.reduce_mean(z8, axis=0, keep_dims=True)
        # z8 = tf.nn.tanh(z8)
        y4_hat = tf.reshape(z8, [-1, 590080])

        w9 = tf.get_variable('conv5-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z9 = tf.nn.conv2d(a4, w9, strides=[1, 1, 1, 1], padding='VALID')
        # z9 = tf.layers.batch_normalization(z9, training=is_train2)
        # z9 = tf.reduce_mean(z9, axis=0, keep_dims=True)
        # z9 = tf.nn.tanh(z9)
        y5_hat = tf.reshape(z9, [-1, 590080])

        w10 = tf.get_variable('conv6-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z10 = tf.nn.conv2d(a4, w10, strides=[1, 1, 1, 1], padding='VALID')
        # z10 = tf.layers.batch_normalization(z10, training=is_train2)
        # z10 = tf.reduce_mean(z10, axis=0, keep_dims=True)
        # z10 = tf.nn.tanh(z10)
        y6_hat = tf.reshape(z10, [-1, 590080])

    # Decoder-2: decode vgg11 fcn layer
    with tf.variable_scope('Decoder-2'):
        w11 = tf.get_variable('fcn1-weight', [flatten.shape[1], 10250],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        z11 = tf.matmul(flatten, w11)
        # z11 = tf.layers.batch_normalization(z11, training=is_train2)
        # z11 = tf.reduce_mean(z11, axis=0, keep_dims=True)
        # z11 = tf.nn.tanh(z11)
        y7_hat = tf.reshape(z11, [-1, 10250])

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

    # Decoder 切片
    # 张量切片: biases = z5[1, 30720:30840]

    decoder_conv_w5 = y3_hat[0, 0: 294912]
    decoder_conv_w5 = tf.reshape(decoder_conv_w5, [3, 3, 128, 256])
    decoder_conv_b5 = y3_hat[0, 294912: 295168]

    decoder_conv_w6 = y4_hat[0, 0: 589824]
    decoder_conv_w6 = tf.reshape(decoder_conv_w6, [3, 3, 256, 256])
    decoder_conv_b6 = y4_hat[0, 589824: 590080]

    decoder_conv_w7 = y5_hat[0, 0: 589824]
    decoder_conv_w7 = tf.reshape(decoder_conv_w7, [3, 3, 256, 256])
    decoder_conv_b7 = y5_hat[0, 589824: 590080]

    decoder_conv_w8 = y6_hat[0, 0: 589824]
    decoder_conv_w8 = tf.reshape(decoder_conv_w8, [3, 3, 256, 256])
    decoder_conv_b8 = y6_hat[0, 589824: 590080]

    decoder_fcn_w9 = y7_hat[0, 0: 10240]
    decoder_fcn_w9 = tf.reshape(decoder_fcn_w9, [1024, 10])
    decoder_fcn_b9 = y7_hat[0, 10240: 10250]

    parameters = {'w1': eval_w1, 'w2': eval_w2, 'w3': eval_w3, 'w4': eval_w4, 'w5': decoder_conv_w5,
                  'w6': decoder_conv_w6,
                  'w7': decoder_conv_w7, 'w8': decoder_conv_w8, 'w9': decoder_fcn_w9, 'b1': eval_b1, 'b2': eval_b2,
                  'b3': eval_b3,
                  'b4': eval_b4, 'b5': decoder_conv_b5, 'b6': decoder_conv_b6, 'b7': decoder_conv_b7,
                  'b8': decoder_conv_b8,
                  'b9': decoder_fcn_b9}

    return parameters