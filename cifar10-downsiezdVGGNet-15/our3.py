import time

import tensorflow as tf
import pickle as pk
import numpy as np
from input_data import Cifar10
import math
from mixup import mixup_data
from seblock import SE_block
import random

# 正则化系数
REGULARIZERATION = 0.0002


def model_parameters_dataset(batch_size):

    with open('datasets2/train_xs', 'rb') as f:
        data = pk.load(f)

    train_data = data
    xs = []
    m = train_data.shape[0]
    nums_batches = m // batch_size
    for k in range(0, nums_batches):
        train_xs = train_data[batch_size * k: batch_size * (k + 1)]
        xs.append(train_xs)

    print("Complete dataset loading! The shape of dataset is {}. Start the \"Predict Network\" training!"
          .format(data.shape))

    return xs, nums_batches


def model_parameters_dataset2(batch_size):
    with open('dataset/xs_layers_1234', 'rb') as f:
        input = pk.load(f)

    # with open('./dataset/y1', 'rb') as f:
    #     y1 = pk.load(f)
    # with open('./dataset/y2', 'rb') as f:
    #     y2 = pk.load(f)
    with open('./dataset/y5', 'rb') as f:
        y5 = pk.load(f)
    with open('./dataset/y6', 'rb') as f:
        y6 = pk.load(f)
    with open('./dataset/y7', 'rb') as f:
        y7 = pk.load(f)
    with open('./dataset/y8', 'rb') as f:
        y8 = pk.load(f)
    with open('./dataset/y9', 'rb') as f:
        y9 = pk.load(f)

    xs = []
    m = input.shape[0]
    nums_batches = m // batch_size
    print(input.shape)
    print(nums_batches)
    for k in range(0, nums_batches):
        train_xs = input[batch_size * k: batch_size * (k + 1)]
        # ys1 = y1[batch_size * k: batch_size * (k + 1)]
        # ys2 = y2[batch_size * k: batch_size * (k + 1)]
        ys5 = y5[batch_size * k: batch_size * (k + 1)]
        ys6 = y6[batch_size * k: batch_size * (k + 1)]
        ys7 = y7[batch_size * k: batch_size * (k + 1)]
        ys8 = y8[batch_size * k: batch_size * (k + 1)]
        ys9 = y9[batch_size * k: batch_size * (k + 1)]

        xs.append((train_xs, ys5, ys6, ys7, ys8, ys9))

    return xs, nums_batches


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
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


def load_dataset():

    np.random.seed(1)
    cifar10 = Cifar10(path=r"cifar-10-batches-py", one_hot=True)
    cifar10._load_data()

    # 训练集
    train_xs = cifar10.images / 255.0
    train_labels = cifar10.labels
    print(train_xs.shape)

    # 准备测试集
    test_xs = cifar10.test.images / 255.0
    test_labels = cifar10.test.labels
    print(test_xs.shape)

    return train_xs, train_labels, test_xs, test_labels


def create_placeholder():

    # weights samples
    x = tf.placeholder(tf.float32, [None, 8, 8, 4065], name='input_x')

    # label
    # y1 = tf.placeholder(tf.float32, [None, 73856], name='label_1')
    # y2 = tf.placeholder(tf.float32, [None, 147584], name='label_2')
    y5 = tf.placeholder(tf.float32, [None, 295168], name='label_3')
    y6 = tf.placeholder(tf.float32, [None, 590080], name='label_4')
    y7 = tf.placeholder(tf.float32, [None, 590080], name='label_5')
    y8 = tf.placeholder(tf.float32, [None, 590080], name='label_6')
    y9 = tf.placeholder(tf.float32, [None, 10250], name='label_7')

    # images
    images_input = tf.placeholder(tf.float32, [None, 32, 32, 3], name='images_input')
    images_label_a = tf.placeholder(tf.float32, [None, 10], name='y_a')
    images_label_b = tf.placeholder(tf.float32, [None, 10], name='y_b')
    # mixup hyper-parameters
    lam_placeholder = tf.placeholder(tf.float32, name='lam')

    # weight hyper-parameters
    beta = tf.placeholder(tf.float32, name='beta')
    random_seed = tf.placeholder(tf.int32, name='random_seed')

    return x, images_input, images_label_a, images_label_b, lam_placeholder, \
           y5, y6, y7, y8, y9, random_seed, beta


def forward_propagation(input_tensor, input_images, seed):

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
        w4 = tf.get_variable('fcn-weight-1', [flatten.shape[1], 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b4 = tf.get_variable('fcn-biases-1', [64], initializer=tf.constant_initializer(0.1))
        z4 = tf.matmul(flatten, w4) + b4
        # z4 = tf.layers.batch_normalization(z4, training=is_train2)
        a4 = tf.nn.relu(z4)
        a4 = tf.reshape(a4, [-1, 8, 8, 1])

        w5 = tf.get_variable('conv5-weight', [1, 1, 1, 4612], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z5 = tf.nn.conv2d(a4, w5, strides=[1, 1, 1, 1], padding='VALID')
        y5_hat = tf.reshape(z5, [-1, 295168])

        w6 = tf.get_variable('conv6-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z6 = tf.nn.conv2d(a4, w6, strides=[1, 1, 1, 1], padding='VALID')
        y6_hat = tf.reshape(z6, [-1, 590080])

        w7 = tf.get_variable('conv7-weight', [1, 1, 1, 9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z7 = tf.nn.conv2d(a4, w7, strides=[1, 1, 1, 1], padding='VALID')
        y7_hat = tf.reshape(z7, [-1, 590080])

        w8 = tf.get_variable('conv8-weight', [1, 1, 1,9220], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z8 = tf.nn.conv2d(a4, w8, strides=[1, 1, 1, 1], padding='VALID')
        y8_hat = tf.reshape(z8, [-1, 590080])

    # Decoder-2: decode vgg11 fcn layer
    with tf.variable_scope('Decoder-2'):
        w9 = tf.get_variable('fcn1-weight', [flatten.shape[1], 10250], initializer=tf.truncated_normal_initializer(stddev=0.1))
        z9 = tf.matmul(flatten, w9)
        y9_hat = tf.reshape(z9, [-1, 10250])

    # 输入切片
    # input_tensor = tf.reduce_mean(input_tensor, axis=0, keep_dims=True)
    input_tensor = tf.reshape(input_tensor, [-1, 260160])

    eval_w1 = input_tensor[seed, 0: 1728]
    eval_w1 = tf.reshape(eval_w1, [3, 3, 3, 64])
    eval_b1 = input_tensor[seed, 1728: 1792]

    eval_w2 = input_tensor[seed, 1792: 38656]
    eval_w2 = tf.reshape(eval_w2, [3, 3, 64, 64])
    eval_b2 = input_tensor[seed, 38656: 38720]

    eval_w3 = input_tensor[seed, 38720: 112448]
    eval_w3 = tf.reshape(eval_w3, [3, 3, 64, 128])
    eval_b3 = input_tensor[seed, 112448: 112576]

    eval_w4 = input_tensor[seed, 112576: 260032]
    eval_w4 = tf.reshape(eval_w4, [3, 3, 128, 128])
    eval_b4 = input_tensor[seed, 260032: 260160]

    # Decoder 切片
    # 张量切片: biases = z5[1, 30720:30840]

    # decoder_conv_w1 = y1_hat[seed, 0: 73728]
    # decoder_conv_w1 = tf.reshape(decoder_conv_w1, [3, 3, 64, 128])
    # decoder_conv_b1 = y1_hat[seed, 73728:73856]

    # decoder_conv_w2 = y2_hat[seed, 0: 147456]
    # decoder_conv_w2 = tf.reshape(decoder_conv_w2, [3, 3, 128, 128])
    # decoder_conv_b2 = y2_hat[seed, 147456: 147584]

    decoder_conv_w5 = y5_hat[seed, 0: 294912]
    decoder_conv_w5 = tf.reshape(decoder_conv_w5, [3, 3, 128, 256])
    decoder_conv_b5 = y5_hat[seed, 294912: 295168]

    decoder_conv_w6 = y6_hat[seed, 0: 589824]
    decoder_conv_w6 = tf.reshape(decoder_conv_w6, [3, 3, 256, 256])
    decoder_conv_b6 = y6_hat[seed, 589824: 590080]

    decoder_conv_w7 = y7_hat[seed, 0: 589824]
    decoder_conv_w7 = tf.reshape(decoder_conv_w7, [3, 3, 256, 256])
    decoder_conv_b7 = y7_hat[seed, 589824: 590080]

    decoder_conv_w8 = y8_hat[seed, 0: 589824]
    decoder_conv_w8 = tf.reshape(decoder_conv_w8, [3, 3, 256, 256])
    decoder_conv_b8 = y8_hat[seed, 589824: 590080]

    decoder_fcn_w9 = y9_hat[seed, 0: 10240]
    decoder_fcn_w9 = tf.reshape(decoder_fcn_w9, [1024, 10])
    decoder_fcn_b9 = y9_hat[seed, 10240: 10250]

    # 2-evaluate network
    # evaluate CIFAR10 images with the predicted weight parameters

    eval_conv1 = tf.nn.conv2d(input_images, eval_w1, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv1 = tf.nn.bias_add(eval_conv1, eval_b1)
    eval_relu1 = tf.nn.relu(eval_conv1)

    eval_conv2 = tf.nn.conv2d(eval_relu1, eval_w2, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv2 = tf.nn.bias_add(eval_conv2, eval_b2)
    eval_relu2 = tf.nn.relu(eval_conv2)

    eval_pool1 = tf.nn.max_pool(eval_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    eval_conv3 = tf.nn.conv2d(eval_pool1, eval_w3, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv3 = tf.nn.bias_add(eval_conv3, eval_b3)
    eval_relu3 = tf.nn.relu(eval_conv3)

    eval_conv4 = tf.nn.conv2d(eval_relu3, eval_w4, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv4 = tf.nn.bias_add(eval_conv4, eval_b4)
    eval_relu4 = tf.nn.relu(eval_conv4)

    eval_pool2 = tf.nn.max_pool(eval_relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    eval_conv5 = tf.nn.conv2d(eval_pool2, decoder_conv_w5, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv5 = tf.nn.bias_add(eval_conv5, decoder_conv_b5)
    eval_relu5 = tf.nn.relu(eval_conv5)

    eval_conv6 = tf.nn.conv2d(eval_relu5, decoder_conv_w6, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv6 = tf.nn.bias_add(eval_conv6, decoder_conv_b6)
    eval_relu6 = tf.nn.relu(eval_conv6)

    eval_pool3 = tf.nn.max_pool(eval_relu6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    eval_conv7 = tf.nn.conv2d(eval_pool3, decoder_conv_w7, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv7 = tf.nn.bias_add(eval_conv7, decoder_conv_b7)
    eval_relu7 = tf.nn.relu(eval_conv7)

    eval_conv8 = tf.nn.conv2d(eval_relu7, decoder_conv_w8, strides=[1, 1, 1, 1], padding='SAME')
    eval_conv8 = tf.nn.bias_add(eval_conv8, decoder_conv_b8)
    eval_relu8 = tf.nn.relu(eval_conv8)

    eval_pool4 = tf.nn.max_pool(eval_relu8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    p = tf.contrib.layers.flatten(eval_pool4)

    logits = tf.matmul(p, decoder_fcn_w9) + decoder_fcn_b9

    # y1_hat = y1_hat[seed, 0: 73856]
    # y1_hat = tf.reshape(y1_hat, [1, -1])

    # y2_hat = y2_hat[seed, 0: 147584]
    # y2_hat = tf.reshape(y2_hat, [1, -1])

    y5_hat = y5_hat[seed, 0: 295168]
    y5_hat = tf.reshape(y5_hat, [1, -1])

    y6_hat = y6_hat[seed, 0: 590080]
    y6_hat = tf.reshape(y6_hat, [1, -1])

    y7_hat = y7_hat[seed, 0: 590080]
    y7_hat = tf.reshape(y7_hat, [1, -1])

    y8_hat = y8_hat[seed, 0: 590080]
    y8_hat = tf.reshape(y8_hat, [1, -1])

    y9_hat = y9_hat[seed, 0: 10250]
    y9_hat = tf.reshape(y9_hat, [1, -1])

    return logits, is_train, y5_hat, y6_hat, y7_hat, y8_hat, y9_hat


def compute_cost(logits, y_a, y_b, lam, lr, y5, y6, y7, y8, y9,
                 y5_hat, y6_hat, y7_hat, y8_hat, y9_hat, beta, trainable_vars):
    # evaluation loss
    cross_entropy_a = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(y_a, 1))
    cross_entropy_b = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(y_b, 1))
    loss_a = tf.reduce_mean(cross_entropy_a)
    loss_b = tf.reduce_mean(cross_entropy_b)

    # weight loss
    w_loss = tf.reduce_mean(tf.square(y5 - y5_hat)) + tf.reduce_mean(tf.square(y6 - y6_hat)) + \
             tf.reduce_mean(tf.square(y7 - y7_hat)) + tf.reduce_mean(tf.square(y8 - y8_hat)) + \
             tf.reduce_mean(tf.square(y9 - y9_hat))

    # beta hyper-parameters
    loss1 = w_loss
    loss2 = loss_a * lam + loss_b * (1 - lam)
    # 1、总是取最小损失
    # loss = tf.math.maximum(loss1, loss2)
    # 2、加权的方式
    loss = loss1*beta + loss2*(1 - beta)
    # loss = loss_a*lam + loss_b*(1 - lam) + beta*w_loss
    opt = tf.train.AdamOptimizer(lr)
    grads_and_vars = opt.compute_gradients(loss, var_list=trainable_vars)
    update_opt = opt.apply_gradients(grads_and_vars)
    # optimizer = tf.train.AdamOptimizer(lr).minimize(loss, var_list=trainable_vars)

    return logits, loss, opt, update_opt, grads_and_vars


def compute_accuracy(logits, y):

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def batch_normalization(input, beta, gama, eps=1e-5):

    input_shape = input.get_shape()
    axis = list(range(len(input_shape) - 1))
    mean, var = tf.nn.moments(input, axis)

    return tf.nn.batch_normalization(input, mean, var, beta, gama, eps)


def main(epoch_nums):

    input_paras, nums_xs_batch = model_parameters_dataset2(batch_size=200)

    predict_input, eval_input, eval_ya, eval_yb, lam_tensor, y5, y6, y7, y8, y9, random_seed, beta = create_placeholder()
    # 加入正则化
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZERATION)
    logits, istrain, y5_hat, y6_hat, y7_hat, y8_hat, y9_hat = forward_propagation(predict_input, eval_input, random_seed)

    variables = tf.trainable_variables()
    # trainable_vars = variables[20:]
    # print('trainable_vars:')
    # print(trainable_vars)
    # untrained_vars = variables[6:20]

    # print('untrained_vars:')
    # print(untrained_vars)

    logits, loss,  opt, update_opt, grads_and_vars = compute_cost(logits, eval_ya, eval_yb, lam_tensor, 0.001,
                                          y5, y6, y7, y8, y9, y5_hat, y6_hat, y7_hat, y8_hat, y9_hat, beta, variables)
    accuracy = compute_accuracy(logits, eval_ya)

    # trainable_vars = tf.trainable_variables()
    # print(trainable_vars)

    init = tf.global_variables_initializer()
    # saver1 = tf.train.Saver(untrained_vars)
    saver2 = tf.train.Saver()

    train_xs, train_ys, test_xs, test_ys = load_dataset()
    nums_image_batch = int(train_xs.shape[0] / 125)
    l = list(np.arange(start=0, stop=nums_image_batch, step=1))

    costs = []
    test_accs = []

    max_acc = 0
    iterations = 0

    with tf.Session() as sess:
        sess.run(init)
        # saver1.restore(sess, './model-resnet/pred_model_for_resnet/model.ckpt')      # 重新加载模型，微调后继续训练

        for epoch in range(1, epoch_nums+1):
            epoch_cost = 0
            sum_of_time = 0
            temp = []
            for xs in input_paras:
                start = time.time()
                seed = np.random.randint(0, 200)
                (X, Y5, Y6, Y7, Y8, Y9) = xs

                Y5 = Y5[seed, 0: 295168]
                Y5 = Y5.reshape(1, -1)

                Y6 = Y6[seed, 0: 590080]
                Y6 = Y6.reshape(1, -1)

                Y7 = Y7[seed, 0: 590080]
                Y7 = Y7.reshape(1, -1)

                Y8 = Y8[seed, 0: 590080]
                Y8 = Y8.reshape(1, -1)

                Y9 = Y9[seed, 0: 10250]
                Y9 = Y9.reshape(1, -1)
                X = X.reshape([-1, 8, 8, 4065])
                iterations_cost = 0
                iterations += 1
                mini_batches = random_mini_batches(train_xs, train_ys, 125, seed=epoch)
                random_list = random.sample(l, 40)
                for index in random_list:
                    (mini_batch_X, mini_batch_Y) = mini_batches[index]
                    # mixup : 混合增强
                    x, mix_x, label_a, label_b, lam = mixup_data(mini_batch_X, mini_batch_Y, alpha=1)
                    cost, update, grads_and_vars_list, train_acc = sess.run([loss, update_opt, grads_and_vars, accuracy],
                                                  feed_dict={predict_input: X, eval_input: mix_x,
                                                             eval_ya: label_a, eval_yb: label_b,
                                                             y5: Y5, y6: Y6, y7: Y7, y8: Y8, y9: Y9,
                                                             lam_tensor: lam, beta: 0.3, random_seed: seed
                                                             })
                    iterations_cost += cost/40

                end = time.time()
                sum_of_time += (end - start)
                test_acc = sess.run(accuracy, feed_dict={predict_input: X, eval_input: test_xs, eval_ya: test_ys,
                                                         random_seed: seed})
                temp.append(test_acc)

                if test_acc > max_acc:
                    max_acc = test_acc
                    saver2.save(sess, "dataset/save_model1/model.ckpt")    # 保存精度最高时候的模型
                if iterations % 1 == 0:
                    print("Epoch {}/{}, Iteration {}/{}, Training cost is {}, Test accuracy is {}"
                          .format(epoch, epoch_nums, iterations, epoch_nums * nums_xs_batch, iterations_cost, test_acc))
                epoch_cost += iterations_cost/nums_xs_batch

            test_accs.append(max(temp))
            costs.append(epoch_cost)
            if epoch % 1 == 0:
                print("After {} Epoch, epoch cost is {}, Max test accuracy is {}, Test accuracy is {}, {} sec/batch"
                      .format(epoch, epoch_cost, max_acc, test_accs[-1], sum_of_time//nums_xs_batch))
        # 保存收敛时的模型
        # saver.save(sess, './save_model/model.ckpt')

    with open('dataset/result1/cost', 'wb') as f:
        f.write(pk.dumps(costs))
    with open('dataset/result1/test_accs', 'wb') as f:
        f.write(pk.dumps(test_accs))


if __name__=='__main__':
    main(epoch_nums=30)

