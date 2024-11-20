
import tensorflow as tf
from socket import *
import numpy as np
import pickle as pk
import math
import time
import seblock
from downsized_vgg15_init import create_placeholder
from mixup import mixup_data
from input_data import Cifar10


# 配置神经网络的参数
INPUT_NODE = 3072
OUTPUT_NODE = 10

# 输入图片的大小
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 64
CONV1_SIZE = 3

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 3

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 128
CONV3_SIZE = 3

# 第四层卷积层的尺寸和深度
CONV4_DEEP = 128
CONV4_SIZE = 3

# 第五层卷积层的尺寸和深度
CONV5_DEEP = 256
CONV5_SIZE = 3

# 第六层卷积层的尺寸和深度
CONV6_DEEP = 256
CONV6_SIZE = 3

# 第7层卷积层的尺寸和深度
CONV7_DEEP = 256
CONV7_SIZE = 3

# 第8层卷积层的尺寸和深度
CONV8_DEEP = 256
CONV8_SIZE = 3

# 第9层卷积层的尺寸和深度
CONV9_DEEP = 256
CONV9_SIZE = 3

# 第10层卷积层的尺寸和深度
CONV10_DEEP = 256
CONV10_SIZE = 3

# 第11层卷积层的尺寸和深度
CONV11_DEEP = 256
CONV11_SIZE = 3

# 第12层卷积层的尺寸和深度
CONV12_DEEP = 256
CONV12_SIZE = 3

# 第13层卷积层的尺寸和深度
CONV13_DEEP = 256
CONV13_SIZE = 3

# 第14层全连接层的尺寸和深度
FC1_SIZE = 100
# 第15层全连接层的尺寸和深度
FC2_SIZE = 100

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.0001, """Learning rate.""")
tf.app.flags.DEFINE_integer('mini_batch_size', 1000, """Number of mini batch.""")
tf.app.flags.DEFINE_integer('worker_batch_size', 125, """The number of sample for each worker to train.""")
tf.app.flags.DEFINE_integer('partition', 0, """Data partition.""")
tf.app.flags.DEFINE_float('delta_acc', 0.002, """The convergence condition.""")
tf.app.flags.DEFINE_integer('len', 37783269, """Number of parameters byte.""")
tf.app.flags.DEFINE_integer('total_epochs', 300, 'Total number of epochs')

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2223, '''The port of parameter server''')
tf.app.flags.DEFINE_string('ip_address', '172.26.240.103', '''The ip address of parameter server''')


def forward_propagation(input_tensor):
    is_train = tf.placeholder_with_default(False, (), 'is_train')
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, conv1_biases)
        conv1 = tf.layers.batch_normalization(conv1, training=is_train)
        relu1 = tf.nn.relu(conv1)
        relu1 = seblock.SE_block(relu1, 4)

    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, conv2_biases)
        conv2 = tf.layers.batch_normalization(conv2, training=is_train)
        relu2 = tf.nn.relu(conv2)
        relu2 = seblock.SE_block(relu2, 4)

    with tf.variable_scope('pool1'):
        pool1 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool1, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.bias_add(conv3, conv3_biases)
        conv3 = tf.layers.batch_normalization(conv3, training=is_train)
        relu3 = tf.nn.relu(conv3)
        relu3 = seblock.SE_block(relu3, 4)

    with tf.variable_scope('layer4-conv4'):
        conv4_weights = tf.get_variable("weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = tf.nn.bias_add(conv4, conv4_biases)
        conv4 = tf.layers.batch_normalization(conv4, training=is_train)
        relu4 = tf.nn.relu(conv4)
        relu4 = seblock.SE_block(relu4, 4)

    with tf.variable_scope('pool2'):
        pool2 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer5-conv5'):
        conv5_weights = tf.get_variable("weight", [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool2, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = tf.nn.bias_add(conv5, conv5_biases)
        conv5 = tf.layers.batch_normalization(conv5, training=is_train)
        relu5 = tf.nn.relu(conv5)
        relu5 = seblock.SE_block(relu5, 4)

    with tf.variable_scope('layer6-conv6'):
        conv6_weights = tf.get_variable("weight", [CONV6_SIZE, CONV6_SIZE, CONV5_DEEP, CONV6_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv6_biases = tf.get_variable("bias", [CONV6_DEEP], initializer=tf.constant_initializer(0.0))
        conv6 = tf.nn.conv2d(relu5, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv6 = tf.nn.bias_add(conv6, conv6_biases)
        conv6 = tf.layers.batch_normalization(conv6, training=is_train)
        relu6 = tf.nn.relu(conv6)
        relu6 = seblock.SE_block(relu6, 4)

    with tf.variable_scope('layer7-conv7'):
        conv7_weights = tf.get_variable("weight", [CONV7_SIZE, CONV7_SIZE, CONV6_DEEP, CONV7_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv7_biases = tf.get_variable("bias", [CONV7_DEEP], initializer=tf.constant_initializer(0.0))
        conv7 = tf.nn.conv2d(relu6, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv7 = tf.nn.bias_add(conv7, conv7_biases)
        conv7 = tf.layers.batch_normalization(conv7, training=is_train)
        relu7 = tf.nn.relu(conv7)
        relu7 = seblock.SE_block(relu7, 4)

    with tf.variable_scope('pool3'):
        pool3 = tf.nn.max_pool(relu7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer8-conv8'):
        conv8_weights = tf.get_variable("weight", [CONV8_SIZE, CONV8_SIZE, CONV7_DEEP, CONV8_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv8_biases = tf.get_variable("bias", [CONV8_DEEP], initializer=tf.constant_initializer(0.0))
        conv8 = tf.nn.conv2d(pool3, conv8_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv8 = tf.nn.bias_add(conv8, conv8_biases)
        conv8 = tf.layers.batch_normalization(conv8, training=is_train)
        relu8 = tf.nn.relu(conv8)
        relu8 = seblock.SE_block(relu8, 4)

    with tf.variable_scope('layer9-conv9'):
        conv9_weights = tf.get_variable("weight", [CONV9_SIZE, CONV9_SIZE, CONV8_DEEP, CONV9_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv9_biases = tf.get_variable("bias", [CONV9_DEEP], initializer=tf.constant_initializer(0.0))
        conv9 = tf.nn.conv2d(relu8, conv9_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv9 = tf.nn.bias_add(conv9, conv9_biases)
        conv9 = tf.layers.batch_normalization(conv9, training=is_train)
        relu9 = tf.nn.relu(conv9)
        relu9 = seblock.SE_block(relu9, 4)

    # with tf.variable_scope('layer10-conv10'):
    #     conv10_weights = tf.get_variable("weight", [CONV10_SIZE, CONV10_SIZE, CONV9_DEEP, CONV10_DEEP],
    #                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     conv10_biases = tf.get_variable("bias", [CONV10_DEEP], initializer=tf.constant_initializer(0.0))
    #     conv10 = tf.nn.conv2d(relu9, conv10_weights, strides=[1, 1, 1, 1], padding='SAME')
    #     conv10 = tf.nn.bias_add(conv10, conv10_biases)
    #     relu10 = tf.nn.relu(conv10)

    with tf.variable_scope('pool4'):
        pool4 = tf.nn.max_pool(relu9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer11-conv11'):
        conv11_weights = tf.get_variable("weight", [CONV11_SIZE, CONV11_SIZE, CONV10_DEEP, CONV11_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv11_biases = tf.get_variable("bias", [CONV11_DEEP], initializer=tf.constant_initializer(0.0))
        conv11 = tf.nn.conv2d(pool4, conv11_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv11 = tf.nn.bias_add(conv11, conv11_biases)
        conv11 = tf.layers.batch_normalization(conv11, training=is_train)
        relu11 = tf.nn.relu(conv11)
        relu11 = seblock.SE_block(relu11, 4)

    with tf.variable_scope('layer12-conv12'):
        conv12_weights = tf.get_variable("weight", [CONV12_SIZE, CONV12_SIZE, CONV11_DEEP, CONV12_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv12_biases = tf.get_variable("bias", [CONV12_DEEP], initializer=tf.constant_initializer(0.0))
        conv12 = tf.nn.conv2d(relu11, conv12_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv12 = tf.nn.bias_add(conv12, conv12_biases)
        conv12 = tf.layers.batch_normalization(conv12, training=is_train)
        relu12 = tf.nn.relu(conv12)
        relu12 = seblock.SE_block(relu12, 4)

    with tf.variable_scope('layer13-conv13'):
        conv13_weights = tf.get_variable("weight", [CONV13_SIZE, CONV13_SIZE, CONV12_DEEP, CONV13_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv13_biases = tf.get_variable("bias", [CONV13_DEEP], initializer=tf.constant_initializer(0.0))
        conv13 = tf.nn.conv2d(relu12, conv13_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv13 = tf.nn.bias_add(conv13, conv13_biases)
        conv13 = tf.layers.batch_normalization(conv13, training=is_train)
        relu13 = tf.nn.relu(conv13)
        relu13 = seblock.SE_block(relu13, 4)

    with tf.variable_scope('pool5'):
        pool5 = tf.nn.max_pool(relu13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    P = tf.contrib.layers.flatten(pool5)

    with tf.variable_scope('layer11-fc1'):
        fc1_weights = tf.get_variable('weight', [P.shape[1], FC1_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable('biases', [FC1_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.matmul(P, fc1_weights) + fc1_biases
        fc1 = tf.layers.batch_normalization(fc1, training=is_train)
        fc1 = tf.nn.relu(fc1)

    with tf.variable_scope('layer12-fc2'):
        fc2_weights = tf.get_variable('weight', [FC1_SIZE, FC2_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable('biases', [FC2_SIZE], initializer=tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases
        fc2 = tf.layers.batch_normalization(fc2, training=is_train)
        fc2 = tf.nn.relu(fc2)

    with tf.variable_scope('layer13-fc3'):
        fc3_weights = tf.get_variable('weight', [FC2_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc3_biases = tf.get_variable('biases', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logits, is_train


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

    nums_each_class = 1000  # cifar-100, 500 images of each class in test set
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

# 其他辅助函数
def convert_dict_to_tuple(parameters_dict):
    dic = parameters_dict
    return (
        dic['w1'], dic['b1'], dic['w2'], dic['b2'], dic['w3'], dic['b3'],
        dic['w4'], dic['b4'], dic['w5'], dic['b5'], dic['w6'], dic['b6'],
        dic['w7'], dic['b7'], dic['w8'], dic['b8'], dic['w9'], dic['b9'],
        dic['w10'], dic['b10'], dic['w11'], dic['b11'], dic['w12'], dic['b12'],
        dic['w13'], dic['b13'], dic['w14'], dic['b14'], dic['w15'], dic['b15']
    )


def replace_trainable_vars(trainable_vars, parameters):
    l = len(parameters)
    replace = []
    for i in range(l):
        assign = tf.assign(trainable_vars[i], parameters[i])
        replace.append(assign)
    return replace


def exponential_decay(epoch):
    lr = FLAGS.init_lr * FLAGS.decay_rate ** (epoch)
    return lr


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
            parameters['w' + str(j + 1)] = paras
        else:
            parameters['b' + str(j + 1)] = paras
        i += 1
    return parameters


def tcp_connection(ip_address, port):

    workersocket = socket(AF_INET, SOCK_STREAM)
    workersocket.connect((ip_address, port))
    print("Connect Success! Worker ready to receive the initial parameters.")

    return workersocket


def close_socket(workersocket):
    workersocket.send(b'0x03')
    workersocket.close()
    print("Socket closed!")


def recv_initial_parameters(workersocket):
    data = b""
    while True:
        pull_initial_parameters = workersocket.recv(2048000000)
        data += pull_initial_parameters
        if len(data) == FLAGS.len:
            break
    parameters = pk.loads(data)
    print("Receive the initial parameters success ! Worker start training !")
    return parameters

def push_parameters_to_server(workersocket, parameters, epoch):
    # Determine the expected length based on the epoch
    ################################################################ 1111111111
    if epoch <= 2:
        expected_length = 19039758
    else:
        expected_length = 18892477

    data = {'parameters': parameters, 'epoch': epoch}
    drumps_parameters = pk.dumps(data)
    # if epoch > 2:
        # print(222)
        # print(len(drumps_parameters))
    workersocket.send(drumps_parameters)  # Send the parameters and epoch to the server

    # Receive new parameters from the server
    received_data = b""
    # if epoch > 5:
    #     print(234)
    while True:
        # if epoch > 5:
        #     print(567)
        new_data = workersocket.recv(2048000000)
        received_data += new_data
        ################################################################   1111
        # if epoch > 2:
        #     print(333)
        #     print(len(received_data))
        if len(received_data) == expected_length:
            break

    # Unpickle the received data
    new_parameters = pk.loads(received_data)
    return new_parameters

def extract_weights(trainable_vars):
    l = 0
    layer = 1
    weights = []
    bn_and_se_vars = []
    for var in trainable_vars:
        if layer <= 12:
            l += 1
            if l < 3:
                weights.append(var)
            else:
                bn_and_se_vars.append(var)
                if l == 8:
                    l = 0
                    layer += 1
        else:
            l += 1
            if l < 3:
                weights.append(var)
            else:
                bn_and_se_vars.append(var)
                if l == 4:
                    l = 0
                    layer += 1
    return weights, bn_and_se_vars


def main():
    workersocket = tcp_connection(FLAGS.ip_address, FLAGS.port)
    X, target_a, target_b, parameters, lam_tensor = create_placeholder()
    logits, is_train = forward_propagation(X)
    trainable_vars = tf.trainable_variables()
    weights, bn_and_se_vars = extract_weights(trainable_vars)
    logits, cost, update_op, lr = compute_cost(logits, target_a, target_b, lam_tensor, FLAGS.lr,
                                               trainable_vars=bn_and_se_vars)
    accuracy = compute_accuracy(logits, target_a)
    train_xs, train_labels, test_feeds = load_dataset()
    init = tf.global_variables_initializer()
    replace = replace_trainable_vars(weights, parameters)
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

    with tf.device('/GPU:0'):
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(1, FLAGS.total_epochs + 1):
                epoch_cost = 0
                com_time = 0
                mini_batches = random_mini_batches(train_xs, train_labels, FLAGS.mini_batch_size, seed=epoch)
                worker_batches = worker_random_minibatches(mini_batches, worker_minibatch_size=FLAGS.worker_batch_size,
                                                           seed=epoch + 1)

                # Adjust update_vars based on the epoch

                ##################################################################     11111
                if epoch <= 2:
                    update_vars = weights[0: 8] + weights[24: 30]
                else:
                    update_vars = weights[0: 30]

                for worker_batch in worker_batches:
                    (worker_batch_X, worker_batch_Y) = worker_batch[FLAGS.partition]
                    x, mix_x, y_a, y_b, lam = mixup_data(worker_batch_X, worker_batch_Y, alpha=1)
                    bn, _, temp_cost, decay_lr, updated, updated_vars, train_acc = sess.run(
                        [tf.get_collection(tf.GraphKeys.UPDATE_OPS), replace, cost, lr, update_op, update_vars,
                         accuracy],
                        feed_dict={X: mix_x, target_a: y_a, target_b: y_b, lam_tensor: lam, parameters: p,
                                   is_train: True}
                    )
                    epoch_cost += temp_cost / mini_batch_num


                    if epoch <= 2:
                        one_dimension_paras = shape_process1(updated_vars)
                    else:
                        one_dimension_paras = shape_process2(updated_vars)

                    com_start = time.time()
                    new_parameters = push_parameters_to_server(workersocket, one_dimension_paras, epoch)

                    com_end = time.time()
                    com_time += (com_end - com_start)
                    p = convert_dict_to_tuple(new_parameters)

                avg_acc = 0
                for test_feed in test_feeds:
                    test_acc, predicts = sess.run([accuracy, logits], feed_dict={X: test_feed[0], target_a: test_feed[1], is_train: False})
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

                delta_acc = abs(avg_acc - prev_acc)
                move_avg_acc = 0.9 * move_avg_acc + 0.1 * delta_acc
                biased_acc = move_avg_acc / (1 - 0.9**epoch)

                if epoch % 1 == 0:
                    print("CIFAR-100: Epoch {}, Worker{}, Loss = {}, Train_acc = {:.4f}, Communication Time = {:.4f} (s), "
                          "Biased_acc = {:.5f}, Learning rate = {:.6f}".
                          format(epoch, FLAGS.partition+1, epoch_cost, train_acc, com_time, biased_acc, decay_lr))

                if epoch % 5 == 0:
                    print("CIFAR-100: Epoch {}, Worker{}, Avg_acc = {:.4f}, Max_acc = {:.4f}, Precision = {:.4f}, "
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

            with open('results2/test_accs', 'wb') as f:
                f.write(pk.dumps(test_accs))
            with open('results2/loss', 'wb') as f:
                f.write(pk.dumps(costs))
            with open('results2/recall', 'wb') as f:
                f.write(pk.dumps(recalls))
            with open('results2/precisions', 'wb') as f:
                f.write(pk.dumps(precisions))
            with open('results2/F1_scores', 'wb') as f:
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
