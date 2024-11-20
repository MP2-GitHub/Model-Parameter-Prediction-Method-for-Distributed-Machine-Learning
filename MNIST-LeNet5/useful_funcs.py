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
    for i in range(len_of_parameters):
        grads['dw' + str(i+1)] = 0
        grads['db' + str(i+1)] = 0

    return grads


def inference(input_tensor):

    # 1-prediction network
    with tf.variable_scope('layer1-Conv1'):
        w1 = tf.get_variable('weight', [1, 1, 2572, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        z1 = tf.nn.bias_add(tf.nn.conv2d(input_tensor, w1, strides=[1, 1, 1, 1], padding='SAME'), b1)
        a1 = tf.nn.relu(z1)
        a1 = SE_block(a1, ratio=4)
        a1 = tf.reshape(a1, [-1, 8, 8, 1])

    # Pool layer
    with tf.variable_scope('pool1-Layer'):
        pool1 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer-2: Conv-(3, 3, 6, 8)
    with tf.variable_scope('layer2-Conv2'):
        w2 = tf.get_variable('weight', [3, 3, 1, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
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

    # Decoder-1: decode vgg11 convolution layer (L3-L8)
    with tf.variable_scope('Decoder-1'):
        decode_fc_w = tf.get_variable('fcn-weight-1', [flatten.shape[1], 64],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        decode_fc_b = tf.get_variable('fcn-biases-1', [64], initializer=tf.constant_initializer(0.1))
        decode_fc_z = tf.matmul(flatten, decode_fc_w) + decode_fc_b
        decode_fc_z = tf.nn.relu(decode_fc_z)

        decode_w3 = tf.get_variable('fcn1-weight', [decode_fc_z.shape[1], 30840],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        decode_z3 = tf.matmul(decode_fc_z, decode_w3)
        y3_hat = tf.reshape(decode_z3, [-1, 30840])

        decode_w4 = tf.get_variable('fcn2-weight', [decode_fc_z.shape[1], 10164],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        decode_z4 = tf.matmul(decode_fc_z, decode_w4)
        y4_hat = tf.reshape(decode_z4, [-1, 10164])

        decode_w5 = tf.get_variable('fcn3-weight', [decode_fc_z.shape[1], 850],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        decode_z5 = tf.matmul(decode_fc_z, decode_w5)
        y5_hat = tf.reshape(decode_z5, [-1, 850])

    input_tensor = tf.reshape(input_tensor, [-1, 2572])

    # 输入切片
    eval_w1 = input_tensor[0, 0: 150]
    eval_w1 = tf.reshape(eval_w1, [5, 5, 1, 6])
    eval_b1 = input_tensor[0, 150: 156]

    eval_w2 = input_tensor[0, 156: 2556]
    eval_w2 = tf.reshape(eval_w2, [5, 5, 6, 16])
    eval_b2 = input_tensor[0, 2556: 2572]

    # Decoder 切片
    decoder_fcn_w1 = y3_hat[0, 0: 30720]
    decoder_fcn_w1 = tf.reshape(decoder_fcn_w1, [256, 120])
    decoder_fcn_b1 = y3_hat[0, 30720:30840]

    decoder_fcn_w2 = y4_hat[0, 0: 10080]
    decoder_fcn_w2 = tf.reshape(decoder_fcn_w2, [120, 84])
    decoder_fcn_b2 = y4_hat[0, 10080:10164]

    decoder_fcn_w3 = y5_hat[0, 0: 840]
    decoder_fcn_w3 = tf.reshape(decoder_fcn_w3, [84, 10])
    decoder_fcn_b3 = y5_hat[0, 840:850]

    parameters = {'w1': eval_w1, 'b1': eval_b1, 'w2': eval_w2, 'b2': eval_b2, 'w3': decoder_fcn_w1,
                  'b3': decoder_fcn_b1, 'w4': decoder_fcn_w2, 'b4': decoder_fcn_b2, 'w5': decoder_fcn_w3,
                  'b5': decoder_fcn_b3}
    return parameters


def create_placeholder():

    xs = tf.placeholder(tf.float32, [None, 2572], name='input_x')

    return xs