"""
code: mp2--worker
model: AlexNet
dataset: cifar10
"""

import tensorflow as tf
from socket import *
import numpy as np
import pickle as pk
import math
from input_data import Cifar10
import time
import seblock
from alexnet_init import create_placeholder
from mixup import mixup_data


# 配置神经网络的参数
INPUT_NODE = 3072
OUTPUT_NODE = 10

# 输入图片的大小
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 96
CONV1_SIZE = 3

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 256
CONV2_SIZE = 5

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 384
CONV3_SIZE = 3

# 第四层卷积层的尺寸和深度
CONV4_DEEP = 384
CONV4_SIZE = 3

# 第五层卷积层的尺寸和深度
CONV5_DEEP = 256
CONV5_SIZE = 3

# 第六层全连接层的尺寸和深度
FC1_SIZE = 100

# 第七层全连接层的尺寸和深度
FC2_SIZE = 100

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.001, """Learning rate.""")
tf.app.flags.DEFINE_integer('mini_batch_size', 1000, """Number of mini batch.""")
tf.app.flags.DEFINE_integer('worker_batch_size', 125, """The number of sample for each worker to train.""")
tf.app.flags.DEFINE_integer('partition', 0, """Data partition.""")
tf.app.flags.DEFINE_float('delta_acc', 0.002, """The convergence condition.""")
tf.app.flags.DEFINE_integer('len', 30014972, """Number of parameters byte.""")


# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 3333, '''The port of parameter server''')
tf.app.flags.DEFINE_string('ip_address', '172.22.106.123', '''The ip address of parameter server''')


def forward_propagation(input_tensor):

    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_biases)
        relu1 = tf.nn.relu(conv1)

    with tf.variable_scope('pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_biases)
        relu2 = tf.nn.relu(conv2)

    with tf.variable_scope('pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.bias_add(conv3, conv3_biases)
        relu3 = tf.nn.relu(conv3)

    with tf.variable_scope('layer4-conv4'):
        conv4_weights = tf.get_variable("weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = tf.nn.bias_add(conv4, conv4_biases)
        relu4 = tf.nn.relu(conv4)

    with tf.variable_scope('layer5-conv5'):
        conv5_weights = tf.get_variable("weight", [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(relu4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = tf.nn.bias_add(conv5, conv5_biases)
        relu5 = tf.nn.relu(conv5)

    with tf.variable_scope('pool3'):
        pool3 = tf.nn.max_pool(relu5, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

    P = tf.contrib.layers.flatten(pool3)

    with tf.variable_scope('layer6-fc1'):
        fc1_weights = tf.get_variable('weight', [P.shape[1], FC1_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable('biases', [FC1_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.matmul(P, fc1_weights) + fc1_biases

    with tf.variable_scope('layer7-fc2'):
        fc2_weights = tf.get_variable('weight', [FC1_SIZE, FC2_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable('biases', [FC2_SIZE], initializer=tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases

    with tf.variable_scope('layer8-fc3'):
        fc3_weights = tf.get_variable('weight', [FC2_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc3_biases = tf.get_variable('biases', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logits


def compute_cost(logits, label_a, label_b, lam, learning_rate, trainable_vars):

    cross_entropy_a = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(label_a, 1))
    cross_entropy_b = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(label_b, 1))
    cross_entropy_mean_a = tf.reduce_mean(cross_entropy_a)
    cross_entropy_mean_b = tf.reduce_mean(cross_entropy_b)
    loss = cross_entropy_mean_a*lam + cross_entropy_mean_b*(1 - lam)
    # 衰减学习率
    step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate=learning_rate, global_step=step, decay_steps=50,
                                    decay_rate=0.999, staircase=True)
    optimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=trainable_vars)
    update_op = optimizer.apply_gradients(grads_and_vars, global_step=step)

    return logits, loss, update_op, lr


def compute_accuracy(logits, labels):
    correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    return accuracy


def statistics(y_hat, y, record, count):

    n = y.shape[0]
    # record = {}               # 正确判为正类
    # count = {}                # 统计所有

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

    return record, count


def compute_metrix(record, count):

    nums_each_class = 1000  # cifar-10, 1000 images of each class in test set
    clsses = 10
    # compute recall
    recalls = {}
    avg_recall = 0
    for k, v in record.items():
        recall = v / nums_each_class
        recalls[k] = recall
        avg_recall += recall / clsses
    # compute precision
    precisions = {}
    avg_pred = 0
    for k in record.keys():
        precision = record[k] / count[k]
        precisions[k] = precision
        avg_pred += precision / clsses

    # compute average F1 score
    F1 = 2 * avg_pred * avg_recall / (avg_pred + avg_recall)

    return avg_recall, avg_pred, F1


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


def load_dataset():

    cifar10 = Cifar10(path=r"./cifar-10-batches-py", one_hot=True)
    cifar10._load_data()

    # 准备训练集
    train_xs = cifar10.images / 255.0  # 归一化
    train_labels = cifar10.labels

    # 准备测试集
    np.random.seed(1)
    permutation = list(np.random.permutation(10000))
    shuffled_tx = cifar10.test.images[permutation, :, :, :] / 255.0
    shuffled_ty = cifar10.test.labels[permutation, :]

    test_feeds = []
    for k in range(10):
        test_xs = shuffled_tx[k * 1000:k * 1000 + 1000, :, :, :]
        test_labels = shuffled_ty[k * 1000:k * 1000 + 1000, :]

        test_feed = (test_xs, test_labels)

        test_feeds.append(test_feed)

    return train_xs, train_labels, test_feeds


def convert_dict_to_tuple(parameters_dict):

    dic = parameters_dict
    tuple = (
             dic['w1'], dic['b1'], dic['w2'], dic['b2'], dic['w3'], dic['b3'],
             dic['w4'], dic['b4'], dic['w5'], dic['b5'], dic['w6'], dic['b6'],
             dic['w7'], dic['b7'], dic['w8'], dic['b8']
            )
    return tuple


def replace_trainable_vars(trainable_vars, parameters):

    l = len(parameters)
    replace = []
    for i in range(l):
        assign = tf.assign(trainable_vars[i], parameters[i])
        replace.append(assign)
    return replace


def exponential_decay(epoch):

    lr = FLAGS.init_lr * FLAGS.decay_rate**(epoch)

    return lr


# def update_parameters(gradients_and_parameters_list, lr, parameters):
#
#     i = 0
#     conv_parameters = np.zeros([1, 1])
#     for grads_and_vars in gradients_and_parameters_list:
#         paras = parameters[i]
#         (g, v) = grads_and_vars
#         paras = paras - lr*g
#         conv_parameters = np.append(conv_parameters, paras.reshape([1, -1]))
#         i += 1
#     conv_parameters = conv_parameters[1:].reshape([1, -1])
#
#     return conv_parameters


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


def tcp_connection(ip_address, port):

    workersocket = socket(AF_INET, SOCK_STREAM)
    workersocket.connect((ip_address, port))
    print("Connect Success! Worker ready to receive the initial parameters.")

    return workersocket


def close_socket(workersocket):
    # 关闭套接字，训练结束
    workersocket.send(b'0x03')
    workersocket.close()
    print("Socket closed!")


def recv_initial_parameters(workersocket):
    data = b""
    while True:
        pull_initial_parameters = workersocket.recv(2048000000)
        data += pull_initial_parameters
        print(111)
        print(len(data))
        if len(data) == FLAGS.len:
            break
    parameters = pk.loads(data)
    print("Receive the initial parameters success ! Worker start training !")

    return parameters


def push_parameters_to_server(workersocket, parameters):
    data = b""
    drumps_parameters = pk.dumps(parameters)
    workersocket.send(drumps_parameters)  # send the grad to server
    print(222)
    print(len(drumps_parameters))

    while True:
        pull_new_parameters = workersocket.recv(2048000000)
        data += pull_new_parameters
        print(333)
        print(len(data))
        if len(data) == 15007956:
            break
    parameters = pk.loads(data)
    return parameters


def main():

    workersocket = tcp_connection(FLAGS.ip_address, FLAGS.port)

    X, target_a, target_b, parameters, lam_tensor = create_placeholder()

    logits = forward_propagation(X)

    trainable_vars = tf.trainable_variables()

    obtain_updated_vars_op = trainable_vars[0: 4]

    logits, cost, update_op, lr = compute_cost(logits, target_a, target_b, lam_tensor, FLAGS.lr, obtain_updated_vars_op)

    accuracy = compute_accuracy(logits, target_a)

    train_xs, train_labels, test_feeds = load_dataset()

    init = tf.global_variables_initializer()

    replace = replace_trainable_vars(trainable_vars, parameters)   # recv initial parameters from server

    init_parameters = recv_initial_parameters(workersocket)

    p = convert_dict_to_tuple(init_parameters)

    saver = tf.train.Saver()

    test_accs = []
    costs = []
    max_acc = 0
    avg_acc = 0
    prev_acc = 0
    biased_acc = 1
    move_avg_acc = 0
    # epoch = 0

    Recall = 0
    Precision = 0
    F1_score = 0

    recalls = []
    precisions = []
    F1_scores = []
    record = {}
    count = {}

    mini_batch_num = int(train_xs.shape[0] / FLAGS.mini_batch_size)

    with tf.Session() as sess:

        sess.run(init)

        # while biased_acc > FLAGS.delta_acc or epoch < 30:
        for epoch in range(1, 301):
            epoch_cost = 0
            # epoch += 1
            com_time = 0
            mini_batches = random_mini_batches(train_xs, train_labels, FLAGS.mini_batch_size, seed=epoch)
            worker_batches = worker_random_minibatches(mini_batches, worker_minibatch_size=FLAGS.worker_batch_size, seed=epoch+1)
            for worker_batch in worker_batches:
                (worker_batch_X, worker_batch_Y) = worker_batch[FLAGS.partition]
                x, mix_x, y_a, y_b, lam = mixup_data(worker_batch_X, worker_batch_Y, alpha=1)
                _, temp_cost, decay_lr, updated, updated_vars, train_acc = \
                    sess.run([replace, cost, lr, update_op, obtain_updated_vars_op, accuracy],
                                                     feed_dict={X: mix_x, target_a: y_a, target_b: y_b, lam_tensor: lam, parameters: p})
                epoch_cost += temp_cost / mini_batch_num
                one_dimension_paras = shape_process1(updated_vars)
                com_start = time.time()
                new_parameters = push_parameters_to_server(workersocket, one_dimension_paras)
                com_end = time.time()
                com_time += (com_end - com_start)
                p = convert_dict_to_tuple(new_parameters)

            for test_feed in test_feeds:
                test_acc, predicts = sess.run([accuracy, logits], feed_dict={X: test_feed[0], target_a: test_feed[1]})
                record, count = statistics(y_hat=predicts, y=test_feed[1], record=record, count=count)
                avg_acc += test_acc / 10

            avg_recall, avg_precision, f1 = compute_metrix(record=record, count=count)
            recalls.append(avg_recall)
            precisions.append(avg_precision)
            F1_scores.append(f1)
            test_accs.append(avg_acc)
            costs.append(epoch_cost)
            record.clear()
            count.clear()
            if avg_acc > max_acc:
                max_acc = avg_acc
                Recall = avg_recall
                Precision = avg_precision
                F1_score = f1
                saver.save(sess, "./save_model/model.ckpt", global_step=epoch)

            # 计算测试集准确率波动
            delta_acc = abs(avg_acc - prev_acc)
            move_avg_acc = 0.9 * move_avg_acc + 0.1 * delta_acc
            biased_acc = move_avg_acc / (1 - 0.9**epoch)

            if epoch % 1 == 0:
                print("Epoch {}, Worker{}, Loss = {}, Train_acc = {:.4f}, Communication Time = {:.4f} (s), "
                      "Biased_acc = {:.5f}, Learning rate = {:.6f}".
                      format(epoch, FLAGS.partition+1, epoch_cost, train_acc, com_time, biased_acc, decay_lr))

            if epoch % 5 == 0:
                print("Epoch {}, Worker{}, Avg_acc = {:.4f}, Max_acc = {:.4f}, Precision = {:.4f}, "
                      "Recall = {:.4f}, F1 score is {:.4f}"
                      .format(epoch, FLAGS.partition + 1, avg_acc, max_acc, Precision, Recall, F1_score))

            prev_acc = avg_acc
            avg_acc = 0
        # close socket
        close_socket(workersocket)
        # load saved model
        model_file = tf.train.latest_checkpoint("./save_model/")
        saver.restore(sess, model_file)

        print("Loads the saved model: ")
        for test_feed in test_feeds:
            test_acc = sess.run(accuracy, feed_dict={X: test_feed[0], target_a: test_feed[1]})
            avg_acc += test_acc / 10
            print("Test accuracy : {:.4f}".format(test_acc))

        print("Average test accuracy is {:.4f}".format(avg_acc))

        with open('./alexnet_results1/test_accs', 'wb') as f:
            f.write(pk.dumps(test_accs))
        with open('./alexnet_results1/loss', 'wb') as f:
            f.write(pk.dumps(costs))
        with open('./alexnet_results1/recall', 'wb') as f:
            f.write(pk.dumps(recalls))
        with open('./alexnet_results1/precisions', 'wb') as f:
            f.write(pk.dumps(precisions))
        with open('./alexnet_results1/F1_scores', 'wb') as f:
            f.write(pk.dumps(F1_scores))


if __name__ == '__main__':
    print('Neural Network Configuration: ')
    print('Learning rate: {}'.format(FLAGS.lr))
    print('Mini_batch_size: {}'.format(FLAGS.mini_batch_size))
    print('Worker_batch_size: {}'.format(FLAGS.worker_batch_size))
    print('Data partition: {}'.format(FLAGS.partition))
    print('The convergence condition: {}'.format(FLAGS.delta_acc))
    print('Number of parameters byte: {}'.format(FLAGS.len))
    print('Network Communication Configuration: ')
    print('The ip address of parameter server: {}'.format(FLAGS.ip_address))
    print('The port of parameter server: {}'.format(FLAGS.port))
    time.sleep(0.5)
    start = time.time()
    main()
    end = time.time()
    run_time = (end - start)/3600
    print("Run time = {:.2f} (h)".format(run_time))










