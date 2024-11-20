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
    x = tf.placeholder(tf.float32, [None, 8, 8, 9646], name='input_x')

    return x


def inference(input_tensor):
    is_train = tf.placeholder_with_default(False, (), 'is_train')
    # 1-prediction network
    # Layer-1: Conv-(3, 3, 1188, 6)
    with tf.variable_scope('layer1-Conv1'):
        w1 = tf.get_variable('weight', [3, 3, 9646, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
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

    # flatten = tf.contrib.layers.flatten(pool3)
    # print(flatten.shape)

    # Decoder-1: decode vgg11 convolution layer (L3-L8)
    with tf.variable_scope('Decoder-1'):
        decode_w = tf.get_variable('decode-weight-1', [flatten.shape[1], 64],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        decode_b = tf.get_variable('decode-bias-1', [64], initializer=tf.constant_initializer(0.1))
        decode_z = tf.matmul(flatten, decode_w) + decode_b
        # z4 = tf.layers.batch_normalization(z4, training=is_train2)
        a4 = tf.nn.relu(decode_z)
        a4 = tf.reshape(a4, [-1, 8, 8, 1])

        w33 = tf.get_variable('conv3-weight', [1, 1, 1, 13830], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z33 = tf.nn.conv2d(a4, w33, strides=[1, 1, 1, 1], padding='VALID')
        y3_hat = tf.reshape(z33, [-1, 885120])  # 3*3*256*384+384

        w4 = tf.get_variable('conv4-weight', [1, 1, 1, 20742], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z4 = tf.nn.conv2d(a4, w4, strides=[1, 1, 1, 1], padding='VALID')
        y4_hat = tf.reshape(z4, [-1, 1327488])  # 3*3*384*384+384

        w5 = tf.get_variable('conv5-weight', [1, 1, 1, 13828], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z5 = tf.nn.conv2d(a4, w5, strides=[1, 1, 1, 1], padding='VALID')
        y5_hat = tf.reshape(z5, [-1, 884992])  # 3*3*384*256+256

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

        w16 = tf.get_variable('fcn3-weight', [decode_fc_z.shape[1], 1010],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        z16 = tf.matmul(decode_fc_z, w16)
        y16_hat = tf.reshape(z16, [-1, 1010])

    # 评估网路
    # 输入切片
    # input_tensor = tf.reduce_mean(input_tensor, axis=0, keep_dims=True)
    input_tensor = tf.reshape(input_tensor, [-1, 617344])

    eval_w1 = input_tensor[0, 0: 2592]  # 3*3*3*96=2592
    eval_w1 = tf.reshape(eval_w1, [3, 3, 3, 96])
    eval_b1 = input_tensor[0, 2592: 2688]  # 3*3*3*96+96=2688

    eval_w2 = input_tensor[0, 2688: 617088]  # 5*5*96*256=614400 + 2688 =617088
    eval_w2 = tf.reshape(eval_w2, [5, 5, 96, 256])
    eval_b2 = input_tensor[0, 617088: 617344]  # 617088+256=617344

    # eval_w3 = input_tensor[seed, 617344: 1502080] # 3*3*256*384=884736 + 617344 =1502080
    # eval_w3 = tf.reshape(eval_w3, [3, 3, 256, 384])
    # eval_b3 = input_tensor[seed, 1502080: 1502464]  # 1502080 + 384 = 1502464

    # Decoder 切片
    # 张量切片: biases = z5[1, 30720:30840]

    decoder_conv_w3 = y3_hat[0, 0: 884736]
    decoder_conv_w3 = tf.reshape(decoder_conv_w3, [3, 3, 256, 384])
    decoder_conv_b3 = y3_hat[0, 884736: 885120]

    decoder_conv_w4 = y4_hat[0, 0: 1327104]
    decoder_conv_w4 = tf.reshape(decoder_conv_w4, [3, 3, 384, 384])
    decoder_conv_b4 = y4_hat[0, 1327104: 1327488]

    decoder_conv_w5 = y5_hat[0, 0: 884736]
    decoder_conv_w5 = tf.reshape(decoder_conv_w5, [3, 3, 384, 256])
    decoder_conv_b5 = y5_hat[0, 884736: 884992]

    decoder_fcn_w1 = y14_hat[0, 0: 25600]
    decoder_fcn_w1 = tf.reshape(decoder_fcn_w1, [256, 100])
    decoder_fcn_b1 = y14_hat[0, 25600: 26700]

    decoder_fcn_w2 = y15_hat[0, 0: 10000]
    decoder_fcn_w2 = tf.reshape(decoder_fcn_w2, [100, 100])
    decoder_fcn_b2 = y15_hat[0, 10000: 10100]

    decoder_fcn_w3 = y16_hat[0, 0: 1000]
    decoder_fcn_w3 = tf.reshape(decoder_fcn_w3, [100, 10])
    decoder_fcn_b3 = y16_hat[0, 1000: 1010]

    parameters = {'w1': eval_w1, 'w2': eval_w2, 'w3': decoder_conv_w3, 'w4': decoder_conv_w4,
                  'w5': decoder_conv_w5, 'w6': decoder_fcn_w1, 'w7': decoder_fcn_w2, 'w8': decoder_fcn_w3,
                  'b1': eval_b1, 'b2': eval_b2, 'b3': decoder_conv_b3, 'b4': decoder_conv_b4,
                  'b5': decoder_conv_b5, 'b6': decoder_fcn_b1, 'b7': decoder_fcn_b2, 'b8': decoder_fcn_b3}

    return parameters