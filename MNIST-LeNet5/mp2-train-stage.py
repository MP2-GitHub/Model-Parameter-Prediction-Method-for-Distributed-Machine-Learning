import tensorflow as tf
import pickle as pk
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math
minst = input_data.read_data_sets("../mnist_data", one_hot=True)
from seblock import SE_block


def model_parameters_dataset(batch_size):

    with open('./paras_dataset/xs_12_lenet5', 'rb') as f:
        data = pk.load(f)

    xs = []
    m = data.shape[0]
    nums_batches = m // batch_size
    for k in range(0, nums_batches):
        train_xs = data[batch_size*k: batch_size*(k+1)]
        xs.append(train_xs)

    print(len(xs))
    print(nums_batches)

    return xs, nums_batches


def load_dataset_mnist(X, Y):
    # 导入MNIST数据集
    reshape_xs = np.reshape(minst.train.images, (minst.train.images.shape[0], 28, 28, 1))
    ys = minst.train.labels

    print(reshape_xs.shape)

    # 准备验证数据
    reshape_validate_xs = np.reshape(minst.validation.images, (minst.validation.images.shape[0], 28, 28, 1))
    validate_ys = minst.validation.labels

    validate_feed = {X: reshape_validate_xs, Y: validate_ys}

    # 准备测试数据
    reshape_test_xs = np.reshape(minst.test.images, (minst.test.images.shape[0], 28, 28, 1))
    test_ys = minst.test.labels

    return reshape_xs, ys, validate_feed, reshape_test_xs, test_ys


def create_placeholder():

    # weights samples
    x = tf.placeholder(tf.float32, [None, 1, 1, 2572], name='input_x')
    # images
    images_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='images_input')
    images_label = tf.placeholder(tf.float32, [None, 10], name='label')

    random_seed = tf.placeholder(tf.int32, name='random_seed')

    return x, images_input, images_label, random_seed


def forward_propagation(input_tensor, input_images, seed):

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
    print(flatten.shape)

    # Decoder-1: decode vgg11 convolution layer (L3-L8)
    with tf.variable_scope('Decoder-1'):

        decode_fc_w = tf.get_variable('fcn-weight-1', [flatten.shape[1], 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        decode_fc_b = tf.get_variable('fcn-biases-1', [64], initializer=tf.constant_initializer(0.1))
        decode_fc_z = tf.matmul(flatten, decode_fc_w) + decode_fc_b
        decode_fc_z = tf.nn.relu(decode_fc_z)

        decode_w3 = tf.get_variable('fcn1-weight', [decode_fc_z.shape[1], 30840], initializer=tf.truncated_normal_initializer(stddev=0.1))
        decode_z3 = tf.matmul(decode_fc_z, decode_w3)
        y3_hat = tf.reshape(decode_z3, [-1, 30840])

        decode_w4 = tf.get_variable('fcn2-weight', [decode_fc_z.shape[1], 10164], initializer=tf.truncated_normal_initializer(stddev=0.1))
        decode_z4 = tf.matmul(decode_fc_z, decode_w4)
        y4_hat = tf.reshape(decode_z4, [-1, 10164])

        decode_w5 = tf.get_variable('fcn3-weight', [decode_fc_z.shape[1], 850], initializer=tf.truncated_normal_initializer(stddev=0.1))
        decode_z5 = tf.matmul(decode_fc_z, decode_w5)
        y5_hat = tf.reshape(decode_z5, [-1, 850])

    input_tensor = tf.reshape(input_tensor, [-1, 2572])

    # 输入切片
    eval_w1 = input_tensor[seed, 0: 150]
    eval_w1 = tf.reshape(eval_w1, [5, 5, 1, 6])
    eval_b1 = input_tensor[seed, 150: 156]

    eval_w2 = input_tensor[seed, 156: 2556]
    eval_w2 = tf.reshape(eval_w2, [5, 5, 6, 16])
    eval_b2 = input_tensor[seed, 2556: 2572]

    # Decoder 切片
    decoder_fcn_w1 = y3_hat[seed, 0: 30720]
    decoder_fcn_w1 = tf.reshape(decoder_fcn_w1, [256, 120])
    decoder_fcn_b1 = y3_hat[seed, 30720:30840]

    decoder_fcn_w2 = y4_hat[seed, 0: 10080]
    decoder_fcn_w2 = tf.reshape(decoder_fcn_w2, [120, 84])
    decoder_fcn_b2 = y4_hat[seed, 10080:10164]

    decoder_fcn_w3 = y5_hat[seed, 0: 840]
    decoder_fcn_w3 = tf.reshape(decoder_fcn_w3, [84, 10])
    decoder_fcn_b3 = y5_hat[seed, 840:850]

    # 2-evaluate network
    # evaluate MNIST images with the predicted weight parameters

    eval_conv1 = tf.nn.conv2d(input_images, eval_w1, strides=[1, 1, 1, 1], padding='VALID')
    eval_relu1 = tf.nn.relu(tf.nn.bias_add(eval_conv1, eval_b1))
    eval_pool1 = tf.nn.max_pool(eval_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    eval_conv2 = tf.nn.conv2d(eval_pool1, eval_w2, strides=[1, 1, 1, 1], padding='VALID')
    eval_relu2 = tf.nn.relu(tf.nn.bias_add(eval_conv2, eval_b2))
    eval_pool2 = tf.nn.max_pool(eval_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    P = tf.contrib.layers.flatten(eval_pool2)

    fc1 = tf.nn.relu(tf.matmul(P, decoder_fcn_w1) + decoder_fcn_b1)
    fc2 = tf.nn.relu(tf.matmul(fc1, decoder_fcn_w2) + decoder_fcn_b2)
    logits = tf.matmul(fc2, decoder_fcn_w3) + decoder_fcn_b3

    return logits


def compute_cost(y_hat, y, lr):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=tf.argmax(y, 1))
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    return y_hat, loss, optimizer


def compute_accuracy(y_hat, y):

    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

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


def main():

    input_paras, nums_xs_batch = model_parameters_dataset(batch_size=50)
    x, images_input, images_label, random_seed = create_placeholder()
    logits = forward_propagation(x, images_input, random_seed)
    y_hat, loss, optimizer = compute_cost(y_hat=logits, y=images_label, lr=0.001)
    accuracy = compute_accuracy(y_hat, images_label)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    reshape_xs, ys, validate_feed, reshape_test_xs, test_ys = load_dataset_mnist(images_input, images_label)
    nums_image_batch = int(minst.train.images.shape[0] / 125)

    costs = []
    test_accs = []
    iterations = 0
    max_acc = 0

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(10):
            epoch_cost = 0
            for xs in input_paras:
                X = xs.reshape([-1, 1, 1, 2572])
                seed = np.random.randint(0, 50)
                iterations_cost = 0
                iterations += 1
                mini_batches = random_mini_batches(reshape_xs, ys, 125, seed=epoch)
                for mini_batch in mini_batches:
                    (mini_batch_X, mini_batch_Y) = mini_batch
                    _, cost, train_acc = sess.run([optimizer, loss, accuracy], feed_dict={x: X,
                                                                     images_input: mini_batch_X,
                                                                     images_label: mini_batch_Y,
                                                                     random_seed: seed})
                    iterations_cost += cost/nums_image_batch
                if iterations % 1 == 0:
                    print("After {} iterations, training cost is {}, training accuracy is {}"
                          .format(iterations,
                                  iterations_cost,
                                  train_acc))
                epoch_cost += iterations_cost/nums_xs_batch
            costs.append(epoch_cost)
            test_acc = sess.run(accuracy, feed_dict={x: X, images_input: reshape_test_xs, images_label: test_ys, random_seed: seed})
            test_accs.append(test_acc)
            if test_acc > max_acc:
                max_acc = test_acc
                saver.save(sess, './save_model/model3/model.ckpt')

            if epoch % 1 == 0:
                print("After {} epoch, training cost is {:.5f}, test accuracy is {:.5f}, max test accuracy is {:.5f}"
                      .format(epoch, epoch_cost, test_acc, max_acc))

    with open('./save_model/model3/cost', 'wb') as f:
        f.write(pk.dumps(costs))
    with open('./save_model/model3/test_accs', 'wb') as f:
        f.write(pk.dumps(test_accs))


if __name__=='__main__':
    main()

