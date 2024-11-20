import numpy as np
import tensorflow as tf
import math
from socket import *
from queue import Queue
import pickle as pk


## TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'


def _relu(x):

    return tf.nn.relu(x)

def _conv(x, filter_size, out_channel, strides, pad='SAME', name='conv'):
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                        tf.float32, initializer=tf.truncated_normal_initializer(
                stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))

        bias = tf.get_variable('bias', [out_channel], initializer=tf.constant_initializer(0.0))
        # if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
        #     tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)
        conv = tf.nn.bias_add(conv, bias)
    return conv

def _fc(x, out_dim, name='fc'):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [x.shape[1], out_dim],
                        tf.float32, initializer=tf.truncated_normal_initializer(
                            stddev=np.sqrt(1.0/out_dim)))
        # regurization = regurizer(w)
        b = tf.get_variable('biases', [out_dim], tf.float32,
                            initializer=tf.constant_initializer(0.1))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)
    return fc


def _bn(x, is_train, name):

    with tf.variable_scope(name):

        bn = tf.layers.batch_normalization(x, training=is_train)

    return bn

## Other helper functions


def separate_grads_and_vars(grad_and_var_list):
    # grads = {}
    xs = []
    ys = []
    i = 0
    k = 0
    j = 0
    for grads_and_vars in grad_and_var_list:

        # grads["dw" + str(j + 1)] = grads_and_vars[0]
        if i < 20:
            if k < 2:
                xs.append(grads_and_vars[1].reshape(1, -1))
            k += 1
            if k % 4 == 0:
                k = 0
        else:
            if j < 2:
                ys.append(grads_and_vars[1].reshape(1, -1))
            j += 1
            if j % 4 == 0:
                j = 0

        i = i + 1
    return xs, ys


def separate_grads_and_vars2(grad_and_var_list):
    # grads = {}
    xs = []
    i = 0
    for grads_and_vars in grad_and_var_list:

        # grads["dw" + str(j + 1)] = grads_and_vars[0]
        if i < 20:
            xs.append(grads_and_vars[1].reshape(1, -1))
        i = i + 1
    return xs


# test
# test = [('dw1', 'w1'), ('db1', 'b1'), ('dgm1', 'gm1'), ('dbt1', 'bt1'),
#         ('dw2', 'w2'), ('db2', 'b2'), ('dgm2', 'gm2'), ('dbt2', 'bt2'),
#         ('dw3', 'w3'), ('db3', 'b3'), ('dgm3', 'gm3'), ('dbt3', 'bt3'),
#         ('dw4', 'w4'), ('db4', 'b4'), ('dgm4', 'gm4'), ('dbt4', 'bt4'),
#         ('dw5', 'w5'), ('db5', 'b5'), ('dgm5', 'gm5'), ('dbt5', 'bt5'),
#         ('dw6', 'w6'), ('db6', 'b6'), ('dgm6', 'gm6'), ('dbt6', 'bt6'),
#         ('dw7', 'w7'), ('db7', 'b7'), ('dgm7', 'gm7'), ('dbt7', 'bt7'),
#         ('dw8', 'w8'), ('db8', 'b8'), ('dgm8', 'gm8'), ('dbt8', 'bt8'),
#         ('dw9', 'w9'), ('db9', 'b9'), ('dgm9', 'gm9'), ('dbt9', 'bt9'),
#         ('dw10', 'w10'), ('db10', 'b10'), ('dgm10', 'gm10'), ('dbt10', 'bt10')]
#
#
# xs, ys = separate_grads_and_vars(test)
#
# print('xs', xs)
# print('ys', ys)

# test
# test = [('dw1', 'w1'), ('db1', 'b1'),
#         ('dw2', 'w2'), ('db2', 'b2'),
#         ('dw3', 'w3'), ('db3', 'b3'),
#         ('dw4', 'w4'), ('db4', 'b4'),
#         ('dw5', 'w5'), ('db5', 'b5'),
#         ('dw6', 'w6'), ('db6', 'b6'),
#         ('dw7', 'w7'), ('db7', 'b7'),
#         ('dw8', 'w8'), ('db8', 'b8'),
#         ('dw9', 'w9'), ('db9', 'b9'),
#         ('dw10', 'w10'), ('db10', 'b10')]
#
#
# xs = separate_grads_and_vars2(test)
#
# print('xs', xs)
# print('ys', ys)


def convert_dict_to_tuple(parameters_dict):

    dic = parameters_dict
    tuple = (
        dic['w1'], dic['b1'], dic['w2'], dic['b2'], dic['w3'], dic['b3'],
        dic['w4'], dic['b4'], dic['w5'], dic['b5'], dic['w6'], dic['b6'],
        dic['w7'], dic['b7'], dic['w8'], dic['b8'], dic['w9'], dic['b9'],
        dic['w10'], dic['b10'], dic['w11'], dic['b11'], dic['w12'], dic['b12'],
        dic['w13'], dic['b13'], dic['w14'], dic['b14'], dic['w15'], dic['b15'],
        dic['w16'], dic['b16'], dic['w17'], dic['b17'], dic['w18'], dic['b18'],
        dic['w19'], dic['b19'], dic['w20'], dic['b20'], dic['w21'], dic['b21'],
        dic['w22'], dic['b22'], dic['w23'], dic['b23'], dic['w24'], dic['b24'],
        dic['w25'], dic['b25'], dic['w26'], dic['b26'], dic['w27'], dic['b27'],
        dic['w28'], dic['b28'], dic['w29'], dic['b29'], dic['w30'], dic['b30'],
        dic['w31'], dic['b31'], dic['w32'], dic['b32'], dic['w33'], dic['b33'],
        dic['w34'], dic['b34'], dic['w35'], dic['b35'], dic['w36'], dic['b36'],
        dic['w37'], dic['b37'], dic['w38'], dic['b38'], dic['w39'], dic['b39'],
        dic['w40'], dic['b40'], dic['w41'], dic['b41'], dic['w42'], dic['b42'],
        dic['w43'], dic['b43'], dic['w44'], dic['b44'], dic['w45'], dic['b45'],
        dic['w46'], dic['b46'], dic['w47'], dic['b47'], dic['w48'], dic['b48'],
        dic['w49'], dic['b49'], dic['w50'], dic['b50'], dic['w51'], dic['b51'],
        dic['w52'], dic['b52'], dic['w53'], dic['b53'], dic['w54'], dic['b54'],
    )
    return tuple


def replace_trainable_vars(trainable_vars, parameters):

    l = len(parameters)
    replace = []
    for i in range(l):
        assign = tf.assign(trainable_vars[i], parameters[i])
        replace.append(assign)
    return replace


def random_mini_batches(X, Y, mini_batch_size = 128, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch &lt; mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def worker_random_minibatches(minibatch, worker_minibatch_size, seed):
    worker_batches = []
    for batch in minibatch:
        batches = random_mini_batches(batch[0], batch[1], worker_minibatch_size, seed)
        worker_batches.append(batches)

    return worker_batches


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

