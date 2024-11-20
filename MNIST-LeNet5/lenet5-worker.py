"""
code: mp2-ps
model: LeNet
dataset: MNIST
"""

import tensorflow as tf
import numpy as np
import math
from socket import *
import pickle as pk
import time
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets("mnist_data", one_hot=True)

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

BATCH_SIZE = 512

REGULARIZERATION = 0.0001

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 6
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 16
CONV2_SIZE = 5

# 第一层全连接层的节点个数
FC1_SIZE = 120

# 第二层全连接层的节点个数
FC2_SIZE = 84

# 第二层池化层拉直后的节点数
POOL2_FLATTEN = 256

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_integer('nums_epoch', 300, """Number of training epoch.""")
tf.app.flags.DEFINE_float('lr', 0.001, """Learning rate.""")
tf.app.flags.DEFINE_integer('mini_batch_size', 1000, """Number of mini batch.""")
tf.app.flags.DEFINE_integer('worker_batch_size', 125, """The number of sample for each worker to train.""")
tf.app.flags.DEFINE_integer('partition', 0, """Data partition.""")
tf.app.flags.DEFINE_integer('init_len', 356022, """Initial Number of parameters byte.""")
tf.app.flags.DEFINE_integer('len', 178318, """Number of parameters byte.""")
# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')
tf.app.flags.DEFINE_string('ip_address', '10.1.2.50', '''The ip address of parameter server''')


def dataset_process(X, Y, minst=minst):
    # 准备训练数据
    # shape = (55000, 784)
    reshape_xs = np.reshape(minst.train.images, (minst.train.images.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    ys = minst.train.labels  # shape = (55000, 10)

    # 准备验证数据
    # shape = (5000, 784)
    reshape_validate_xs = np.reshape(minst.validation.images,
                                     (minst.validation.images.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    validate_ys = minst.validation.labels # shape = (5000, 10)

    validate_feed = {X: reshape_validate_xs, Y: validate_ys}

    # 准备测试数据
    # shape = (10000, 784)
    reshape_test_xs = np.reshape(minst.test.images, (minst.test.images.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    test_ys = minst.test.labels # shape = (10000, 10)
    test_feed = (reshape_test_xs, test_ys)

    return reshape_xs, ys, validate_feed, test_feed


def create_placeholder():

    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='x_input')
    Y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_label')

    conv1_weights = tf.placeholder(tf.float32, [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], name='conv1_weight')
    conv1_biases = tf.placeholder(tf.float32, [CONV1_DEEP], name='conv1_biases')

    conv2_weights = tf.placeholder(tf.float32, [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], name='conv2_weight')
    conv2_biases = tf.placeholder(tf.float32, [CONV2_DEEP], name='conv2_biases')

    fc1_weights = tf.placeholder(tf.float32, [POOL2_FLATTEN, FC1_SIZE], name='fc1_weight')
    fc1_biases = tf.placeholder(tf.float32, [FC1_SIZE], name='fc1_biases')

    fc2_weights = tf.placeholder(tf.float32, [FC1_SIZE, FC2_SIZE], name='fc2_weight')
    fc2_biases = tf.placeholder(tf.float32, [FC2_SIZE], name='fc2_biases')

    fc3_weights = tf.placeholder(tf.float32, [FC2_SIZE, NUM_LABELS], name='fc3_weight')
    fc3_biases = tf.placeholder(tf.float32, [NUM_LABELS], name='fc3_biases')

    paras_holder = (conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights,
     fc1_biases, fc2_weights, fc2_biases, fc3_weights, fc3_biases)

    return X, Y, paras_holder


def forward_propagation(input_tensor, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    P = tf.contrib.layers.flatten(pool2)

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [P.shape[1], FC1_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            regularization_w1 = regularizer(fc1_weights)

        fc1_biases = tf.get_variable('bias', [FC1_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(P, fc1_weights) + fc1_biases)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [FC1_SIZE, FC2_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            regularization_w2 = regularizer(fc2_weights)
        fc2_biases = tf.get_variable('biases', [FC2_SIZE], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable('weight', [FC2_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            regularization_w3 = regularizer(fc3_weights)
        fc3_biases = tf.get_variable('biases', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logits, (regularization_w1 + regularization_w2 + regularization_w3)


def compute_cost(logits, label, learning_rate, regularization, trainable_vars):

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(label, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 加入正则化
    loss = cross_entropy_mean + regularization
    # 衰减学习率
    step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate=learning_rate, global_step=step, decay_steps=50,
                                    decay_rate=0.999, staircase=True)

    optimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(loss=loss, var_list=trainable_vars)
    update_op = optimizer.apply_gradients(grads_and_vars, global_step=step)

    return logits, loss, grads_and_vars, update_op, lr


def compute_accuracy(logits, label):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
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
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch &lt; mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        (last_x, last_y) = mini_batches[-1]
        last_x = np.r_[last_x, mini_batch_X]
        last_y = np.r_[last_y, mini_batch_Y]
        mini_batches[-1] = (last_x, last_y)
        # mini_batch = (mini_batch_X, mini_batch_Y)
        # mini_batches.append(mini_batch)
    return mini_batches


def worker_random_minibatches(minibatches, worker_minibatch_size, seed=1):
    worker_batches = []
    for batch in minibatches:
        batches = random_mini_batches(batch[0], batch[1], worker_minibatch_size, seed)
        worker_batches.append(batches)

    return worker_batches


def update_parameters(gradients_and_parameters_list, lr):

    i = 0
    parameters = {}
    for grads_and_vars in gradients_and_parameters_list:
        (g, v) = grads_and_vars
        v = v - lr*g                       # update
        j = math.floor(i / 2)
        if i % 2 == 0:
            parameters["w" + str(j + 1)] = v
        else:
            parameters["b" + str(j + 1)] = v
        i = i + 1

    return parameters


def tcp_connection(ip_address, port):
    worker1socket = socket(AF_INET, SOCK_STREAM)
    worker1socket.connect((ip_address, port))
    print("Connect Success! Worker ready to receive the initial parameters.")

    return worker1socket


def recv_initial_parameters(workersocket):
    data = b""
    while True:
        pull_initial_parameters = workersocket.recv(2048000000)
        data += pull_initial_parameters
        if len(data) == FLAGS.init_len:
            break
    parameters = pk.loads(data)
    print("Receive the initial parameters success ! Worker start training !")

    return parameters


def push_parameters_to_server(workersocket, paras):
    data = b""
    drumps_paras = pk.dumps(paras)
    workersocket.send(drumps_paras)  # send the grad to server

    while True:
        pull_new_parameters = workersocket.recv(2048000000)
        data += pull_new_parameters
        if len(data) == FLAGS.len:
            break
    parameters = pk.loads(data)
    return parameters


def shape_process1(parameters):

    conv_parameters = np.zeros([1, 1])
    for paras in parameters:
        conv_parameters = np.append(conv_parameters, paras.reshape([1, -1]))
    conv_parameters = conv_parameters[1:].reshape([1, -1])

    return conv_parameters


def shape_process2(parameters_list):

    parameters = {}
    i = 0
    for paras in parameters_list:
        j = i // 2
        if i % 2 == 0:
            parameters['w' + str(j+1)] = paras
        else:
            parameters['b' + str(j+1)] = paras
        i += 1
    return parameters


def compute_metrics(y_hat, y):

    n = y.shape[0]
    record = {}               # 正确判为正类
    count = {}                # 统计所有
    nums_each_class = 1000    # MNIST, 1000 images of each class in test set
    clsses = 10
    for i in range(n):
        index_yhat = np.argmax(y_hat[i])
        index_y = np.argmax(y[i])
        # 记录TP
        if index_yhat == index_y:
            if index_y not in record:
                record[index_y] = 1
            else:
                record[index_y] += 1
        # 计数各类的 TP+FP
        if index_yhat not in count:
            count[index_yhat] = 1
        else:
            count[index_yhat] += 1
    # compute recall
    recalls = {}
    avg_recall = 0
    for k, v in record.items():
        recall = v/nums_each_class
        recalls[k] = recall
        avg_recall += recall/clsses
    # compute precision
    precisions = {}
    avg_pred = 0
    for k in count.keys():
        precision = record[k] / count[k]
        precisions[k] = precision
        avg_pred += precision / clsses

    # compute average F1 score
    F1 = 2*avg_pred*avg_recall/(avg_pred+avg_recall)

    return avg_recall, avg_pred, F1


def replace_option(trainable_vars, paras_holder):

    ops = []
    for i in range(len(trainable_vars)):
        op = tf.assign(trainable_vars[i], paras_holder[i])
        ops.append(op)

    return ops


def convert_dict_to_tuple(parameters_dict):

    dic = parameters_dict
    tuple = (
             dic['w1'], dic['b1'], dic['w2'], dic['b2'], dic['w3'], dic['b3'],
             dic['w4'], dic['b4'], dic['w5'], dic['b5'])
    return tuple


def main():

    workersocket = tcp_connection(FLAGS.ip_address, FLAGS.port)

    X, Y, paras_holder = create_placeholder()

    # 加入正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZERATION)

    logits, regularization = forward_propagation(X, regularizer=regularizer)

    trainable_vars = tf.trainable_variables()

    local_vars = trainable_vars[0:4]

    logits, loss, grads_and_vars, update_op, lr_tensor = compute_cost(logits, Y, FLAGS.lr, regularization, local_vars)

    accuracy = compute_accuracy(logits, Y)

    init = tf.global_variables_initializer()

    # 用来自服务器的模型参数来更新本地的模型参数
    replace_ops = replace_option(trainable_vars, paras_holder)

    paras = recv_initial_parameters(workersocket)

    paras = convert_dict_to_tuple(paras)

    Cost = []
    Validation_acc = []
    test_accs = []

    reshape_xs, ys, validate_feed, test_feed = dataset_process(X, Y, minst)

    num_mini_batch = int(reshape_xs.shape[0] / FLAGS.mini_batch_size)

    saver = tf.train.Saver()

    max_recall = 0
    max_pred = 0
    max_f1 = 0

    max_acc = 0

    recalls = []
    predictions = []
    f1_scores = []

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(FLAGS.nums_epoch):
            # epoch += 1
            epoch_cost = 0
            com_time = 0

            mini_batches = random_mini_batches(reshape_xs, ys, FLAGS.mini_batch_size, seed=epoch)
            worker_batches = worker_random_minibatches(mini_batches, FLAGS.worker_batch_size, seed=epoch+1)

            for mini_batch in worker_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch[FLAGS.partition]    # 分配worker训练的数据集分区
                replace, grads_and_vars_list, _, cost, vars, train_acc, lr = sess.run([replace_ops, grads_and_vars,
                                                                                       update_op, loss, local_vars,
                                                                                       accuracy, lr_tensor],
                             feed_dict={X: mini_batch_X, Y: mini_batch_Y, paras_holder: paras})

                one_dimension_paras = shape_process1(parameters=vars)
                com_start = time.time()
                paras_dict = push_parameters_to_server(workersocket, one_dimension_paras)
                com_end = time.time()
                com_time += (com_end - com_start)
                epoch_cost += cost / (num_mini_batch)
                paras = convert_dict_to_tuple(paras_dict)

            Cost.append(epoch_cost)
            validate_acc = sess.run(accuracy, feed_dict=validate_feed)
            Validation_acc.append(validate_acc)
            test_acc, pred = sess.run([accuracy, logits], feed_dict={X: test_feed[0], Y: test_feed[1]})
            avg_recall, avg_pred, F1 = compute_metrics(pred, test_feed[1])
            test_accs.append(test_acc)
            predictions.append(avg_pred)
            f1_scores.append(F1)
            recalls.append(avg_recall)

            if test_acc > max_acc:
                max_acc = test_acc
                max_recall = avg_recall
                max_pred = avg_pred
                max_f1 = F1
                saver.save(sess, "./save_model/model.ckpt", global_step=epoch)

            if epoch % 1 == 0:
                print("After {} training epochs, learning rate is {:.4f}, training cost is {:.5f}, "
                      "communication time is {:.4f} (sec), "
                      "training accuracy is {:.4f}"
                      .format(epoch, lr, epoch_cost, com_time, train_acc))

            # 每 5 轮输出一次在验证数据集上的测试结果
            if epoch % 5 == 0:
                print("After {} training epochs, validation accuracy is {:.5f}, test accuracy is {:.5f}, "
                      "prediction accuracy is {:.4f}, recall is {:.4f}, F1 score is {:.4f}"
                      .format(epoch, validate_acc, max_acc, max_pred, max_recall, max_f1))

        # 关闭套接字，训练结束
        workersocket.send(b'0x03')
        workersocket.close()
        print("Socket closed!")

        model_file = tf.train.latest_checkpoint("./save_model/")
        saver.restore(sess, model_file)
        test_acc, pred = sess.run([accuracy, logits], feed_dict={X: test_feed[0], Y: test_feed[1]})
        avg_recall, avg_pred, F1 = compute_metrics(pred, test_feed[1])
        print("Test accuracy is {:.5f}".format(test_acc))
        print("Prediction accuracy is {:.5f}".format(avg_pred))
        print("Recall is {:.5f}".format(avg_recall))
        print("F1 score is {:.5f}".format(F1))

    # 将准确度保存为文件
    with open('./results/test_accs', 'wb') as f:
        f.write(pk.dumps(test_accs))
    with open('./results/cost', 'wb') as f:
        f.write(pk.dumps(Cost))
    with open('./results/prediction', 'wb') as f:
        f.write(pk.dumps(predictions))
    with open('./results/recall', 'wb') as f:
        f.write(pk.dumps(recalls))
    with open('./results/f1', 'wb') as f:
        f.write(pk.dumps(f1_scores))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    run_time = (end - start)/3600
    print("Run time is {} (h)".format(run_time))
