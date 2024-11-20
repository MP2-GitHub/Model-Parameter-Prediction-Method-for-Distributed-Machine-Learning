import tensorflow as tf
import numpy as np
import gzip
import os
import pickle as pk
import time
import ResNet
from mixup import mixup_data
from socket import *
import utils
import initializer

# 参数设置
IMAGE_SIZE = 32  # 图像大小调整为 32x32
CHANELS_SIZE = 3  # MNIST 图像通道数调整为 3 (RGB)
OUTPUT_SIZE = 10  # MNIST 类别数

REGULARIZATION = 0.01

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.001, """Learning rate.""")
tf.app.flags.DEFINE_integer('mini_batch_size', 1000, """Number of mini batch.""")
tf.app.flags.DEFINE_integer('worker_batch_size', 125, """The number of sample for each worker to train.""")
tf.app.flags.DEFINE_integer('partition', 0, """Data partition.""")
tf.app.flags.DEFINE_float('MA_delta_acc', 0.002, """The convergence condition.""")
tf.app.flags.DEFINE_integer('len', 25116651, """Size of initial parameters byte.""")
tf.app.flags.DEFINE_integer('nums_epoch', 300, """The number of training epochs.""")

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')
tf.app.flags.DEFINE_string('ip_address', '192.168.1.15', '''The ip address of parameter server''')


def preprocess_mnist_images(images):
    # 调整图像大小从28x28到32x32
    images_resized = tf.image.resize(images, [IMAGE_SIZE, IMAGE_SIZE])
    # 将灰度图像转换为RGB
    images_rgb = tf.image.grayscale_to_rgb(images_resized)
    return images_rgb

def load_dataset():
    def extract_images(filename):
        with gzip.open(filename) as bytestream:
            bytestream.read(16)  # Skip header
            buf = bytestream.read()
            data = np.frombuffer(buf, dtype=np.uint8)
            num_images = data.shape[0] // (28 * 28)
            data = data.reshape(num_images, 28, 28, 1)
            return data

    def extract_labels(filename):
        with gzip.open(filename) as bytestream:
            bytestream.read(8)  # Skip header
            buf = bytestream.read()
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels

    train_images = extract_images('./mnist_data/train-images-idx3-ubyte.gz')
    train_labels = extract_labels('./mnist_data/train-labels-idx1-ubyte.gz')
    test_images = extract_images('./mnist_data/t10k-images-idx3-ubyte.gz')
    test_labels = extract_labels('./mnist_data/t10k-labels-idx1-ubyte.gz')

    # 归一化图像到 [0, 1] 范围
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 预处理图像调整为32x32并转换为RGB
    with tf.Graph().as_default():
        images_tensor = tf.convert_to_tensor(train_images, dtype=tf.float32)
        preprocessed_train_images = preprocess_mnist_images(images_tensor)

        images_tensor = tf.convert_to_tensor(test_images, dtype=tf.float32)
        preprocessed_test_images = preprocess_mnist_images(images_tensor)

        with tf.Session() as sess:
            train_images = sess.run(preprocessed_train_images)
            test_images = sess.run(preprocessed_test_images)

    # 进行One-hot编码
    train_labels = np.eye(OUTPUT_SIZE)[train_labels]
    test_labels = np.eye(OUTPUT_SIZE)[test_labels]

    # 打乱测试集
    np.random.seed(1)
    permutation = np.random.permutation(test_images.shape[0])
    shuffled_test_images = test_images[permutation]
    shuffled_test_labels = test_labels[permutation]

    # 创建测试集批次
    batch_size = 1000
    num_batches = len(shuffled_test_images) // batch_size
    test_feeds = []

    for k in range(num_batches):
        test_xs = shuffled_test_images[k * batch_size: (k + 1) * batch_size]
        test_labels_batch = shuffled_test_labels[k * batch_size: (k + 1) * batch_size]
        test_feed = (test_xs, test_labels_batch)
        test_feeds.append(test_feed)

    return train_images, train_labels, test_feeds



def shape_process(parameters):
    conv_parameters = np.zeros([1, 1])
    for paras in parameters:
        conv_parameters = np.append(conv_parameters, paras.reshape([1, -1]))
    conv_parameters = conv_parameters[1:].reshape([1, -1])
    return conv_parameters


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
        # print(111)
        # print(len(data))
        if len(data) == FLAGS.len:
            break
    parameters = pk.loads(data)
    print("Receive the initial parameters success ! Worker start training !")
    return parameters


def push_parameters_to_server(workersocket, parameters):
    data = b""
    drumps_parameters = pk.dumps(parameters)
    # print(222)
    # print(len(drumps_parameters))
    workersocket.send(drumps_parameters)
    while True:
        pull_new_parameters = workersocket.recv(2048000000)
        data += pull_new_parameters
        # print(333)
        # print(len(data))
        if len(data) == 12559540:
            break
    parameters = pk.loads(data)
    return parameters


def statistics(y_hat, y, record, count):
    n = y.shape[0]
    for i in range(n):
        index_yhat = np.argmax(y_hat[i])
        index_y = np.argmax(y[i])
        if index_yhat == index_y:
            if index_y not in record:
                record[index_y] = 1
            else:
                record[index_y] += 1
        if index_yhat not in count:
            count[index_yhat] = 1
        else:
            count[index_yhat] += 1
    return record, count


def compute_metrix(record, count):
    nums_each_class = 1000  # MNIST, 每类 1000 张图像在测试集
    classes = 10
    recalls = {}
    avg_recall = 0
    for k, v in record.items():
        recall = v / nums_each_class
        recalls[k] = recall
        avg_recall += recall / classes
    precisions = {}
    avg_pred = 0
    for k in record.keys():
        precision = record[k] / count[k]
        precisions[k] = precision
        avg_pred += precision / classes
    F1 = 2 * avg_pred * avg_recall / (avg_pred + avg_recall)
    return avg_recall, avg_pred, F1


def train(num_classes, num_residual_units, k):
    workersocket = tcp_connection(FLAGS.ip_address, FLAGS.port)
    X, y_a, y_b, lam_tensor, parameters = initializer.create_placeholder()
    hp = ResNet.HParams(batch_size=FLAGS.mini_batch_size,
                        num_classes=num_classes,
                        num_residual_units=num_residual_units,
                        k=k,
                        initial_lr=FLAGS.lr)
    network = ResNet.ResNet18(hp=hp, images=X, labels_a=y_a, labels_b=y_b, lam=lam_tensor)
    network.build_model()
    network.build_train_op()
    network.compute_acc()
    network.startup_bn()

    init = tf.global_variables_initializer()
    train_xs, train_ys, test_feeds = load_dataset()
    saver = tf.train.Saver()
    replace = utils.replace_trainable_vars(network.trainable_vars, parameters)
    init_parameters = recv_initial_parameters(workersocket)
    p = utils.convert_dict_to_tuple(init_parameters)
    test_accs = []
    costs = []
    max_acc = 0
    avg_acc = 0
    prev_acc = 0
    MA_delta_acc = 1
    move_avg_acc = 0
    Recall = 0
    Precision = 0
    F1_score = 0
    recalls = []
    precisions = []
    F1_scores = []
    record = {}
    count = {}

    with tf.device("/gpu:0"):
        with tf.Session() as sess:
            seed = 1
            sess.run(init)
            num_mini_batch = int(train_xs.shape[0] / FLAGS.mini_batch_size)
            for epoch in range(1, FLAGS.nums_epoch + 1):
                seed += 1
                epoch_cost = 0
                com_time = 0
                mini_batches = utils.random_mini_batches(train_xs, train_ys, FLAGS.mini_batch_size, seed)
                worker_batches = utils.worker_random_minibatches(mini_batches,
                                                                 worker_minibatch_size=FLAGS.worker_batch_size,
                                                                 seed=epoch + 1)
                for worker_batch in worker_batches:
                    (worker_batch_X, worker_batch_Y) = worker_batch[FLAGS.partition]
                    x, mix_x, target_a, target_b, lam = mixup_data(worker_batch_X, worker_batch_Y, alpha=1)
                    replace_opt, cost, decay_lr, update_opt, train_acc, updated_paras = sess.run(
                        [replace, network.loss, network.lr, network.update, network.acc, network.update_vars_opt],
                        feed_dict={X: mix_x, y_a: target_a, y_b: target_b, lam_tensor: lam, parameters: p})
                    paras_dict = shape_process(updated_paras)
                    com_start = time.time()
                    new_parameters = push_parameters_to_server(workersocket, paras_dict)
                    com_end = time.time()
                    com_time += (com_end - com_start)
                    p = utils.convert_dict_to_tuple(new_parameters)
                    epoch_cost += cost / num_mini_batch
                costs.append(epoch_cost)
                for test_feed in test_feeds:
                    test_acc, y_hat = sess.run([network.acc, network.logits],
                                               feed_dict={X: test_feed[0], y_a: test_feed[1]})
                    record, count = statistics(y_hat=y_hat, y=test_feed[1], record=record, count=count)
                    avg_acc += test_acc / 10
                avg_recall, avg_precision, f1 = compute_metrix(record=record, count=count)
                recalls.append(avg_recall)
                precisions.append(avg_precision)
                F1_scores.append(f1)
                record.clear()
                count.clear()
                if avg_acc > max_acc:
                    max_acc = avg_acc
                    Recall = avg_recall
                    Precision = avg_precision
                    F1_score = f1
                    saver.save(sess, "./save_model/model.ckpt", global_step=epoch)

                test_accs.append(avg_acc)

                delta_acc = abs(avg_acc - prev_acc)
                move_avg_acc = 0.9 * move_avg_acc + 0.1 * delta_acc
                MA_delta_acc = move_avg_acc / (1 - 0.9 ** epoch)

                prev_acc = avg_acc

                if epoch % 1 == 0:
                    print("Epoch {}, Worker{}, Loss = {}, Train_acc = {:.4f}, Communication Time = {:.4f} (s), "
                          "MA_Δacc = {:.5f}, Learning rate = {}".
                          format(epoch, FLAGS.partition + 1, epoch_cost, train_acc, com_time, MA_delta_acc, decay_lr))

                if epoch % 5 == 0:
                    print("Epoch {}, Worker{}, Avg_acc = {:.4f}, Max_acc = {:.4f}, Precision = {:.4f}, "
                          "Recall = {:.4f}, F1 score is {:.4f}"
                          .format(epoch, FLAGS.partition + 1, avg_acc, max_acc, Precision, Recall, F1_score))
                avg_acc = 0
            # close socket
            close_socket(workersocket)
            # load saved model
            model_file = tf.train.latest_checkpoint("./save_model/")
            saver.restore(sess, model_file)

            print("Loads the saved model: ")
            for test_feed in test_feeds:
                test_acc = sess.run(network.acc, feed_dict={X: test_feed[0], y_a: test_feed[1]})
                avg_acc += test_acc / 10
                print("Test accuracy : {:.4f}".format(test_acc))
            print("Average test accuracy is {:.4f}".format(avg_acc))

            with open('./results/costs', 'wb') as f:
                f.write(pk.dumps(costs))
            with open('./results/test_accs', 'wb') as f:
                f.write(pk.dumps(test_accs))
            with open('./results/recall', 'wb') as f:
                f.write(pk.dumps(recalls))
            with open('./results/precisions', 'wb') as f:
                f.write(pk.dumps(precisions))
            with open('./results/F1_scores', 'wb') as f:
                f.write(pk.dumps(F1_scores))


def main():
    train(num_classes=10, num_residual_units=2, k=2)


if __name__ == '__main__':
    print('Neural Network Configuration: ')
    print('Learning rate: {}'.format(FLAGS.lr))
    print('Mini_batch_size: {}'.format(FLAGS.mini_batch_size))
    print('Worker_batch_size: {}'.format(FLAGS.worker_batch_size))
    print('Data partition: {}'.format(FLAGS.partition))
    print('The convergence condition: {}'.format(FLAGS.MA_delta_acc))
    print('Number of epochs {}'.format(FLAGS.nums_epoch))
    print('Number of parameters byte: {}'.format(FLAGS.len))
    print('Network Communication Configuration: ')
    print('The ip address of parameter server: {}'.format(FLAGS.ip_address))
    print('The port of parameter server: {}'.format(FLAGS.port))

    time.sleep(0.5)
    start = time.time()
    main()
    end = time.time()
    run_time = (end - start) / 3600
    print("Run time = {:.2f} (h)".format(run_time))
