"""
code: mp2-worker
model: downsized-resnet-34
dataset: cifar-100
"""

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import ResNet_34
from mixup import mixup_data
from socket import *
import pickle as pk
import utils
import initializer_34
import pickle

# 参数设置
IMAGE_SIZE = 32

CHANELS_SIZE = 3

OUTPUT_SIZE = 100

REGULARIZATION = 0.01

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.001, """Learning rate.""")
tf.app.flags.DEFINE_integer('mini_batch_size', 1000, """Number of mini batch.""")
tf.app.flags.DEFINE_integer('worker_batch_size', 125, """The number of sample for each worker to train.""")
tf.app.flags.DEFINE_integer('partition', 0, """Data partition.""")
tf.app.flags.DEFINE_float('MA_delta_acc', 0.002, """The convergence condition.""")
tf.app.flags.DEFINE_integer('len', 45520372, """Size of initial parameters byte.""")
tf.app.flags.DEFINE_integer('nums_epoch', 2, """The number of training epochs.""")

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')
tf.app.flags.DEFINE_string('ip_address', '192.168.1.6', '''The ip address of parameter server''')

def load_cifar100(path):
    # 载入训练集
    with open(path + '/train', 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')

    # 载入测试集
    with open(path + '/test', 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')

    # 从训练数据中提取特征和标签
    x_train = train_data[b'data']
    y_train = train_data[b'fine_labels']

    # 从测试数据中提取特征和标签
    x_test = test_data[b'data']
    y_test = test_data[b'fine_labels']

    # 归一化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape图像为(32, 32, 3)
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # 将标签转换为独热编码
    y_train = np.eye(100)[y_train]
    y_test = np.eye(100)[y_test]

    # 将测试数据分成10个小批次
    test_feeds = []
    for k in range(10):
        test_xs = x_test[k * 1000:k * 1000 + 1000, :, :, :]
        test_labels = y_test[k * 1000:k * 1000 + 1000, :]
        test_feed = (test_xs, test_labels)
        test_feeds.append(test_feed)
    # test_feeds = (x_test, y_test)

    return x_train, y_train, test_feeds

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
    # print(222)
    # print(len(drumps_parameters))
    workersocket.send(drumps_parameters)  # send the grad to server

    while True:
        pull_new_parameters = workersocket.recv(2048000000)
        data += pull_new_parameters
        # print(333)
        # print(len(data))
        if len(data) == 22762575:
            break
    parameters = pk.loads(data)
    return parameters


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

    nums_each_class = 100  # cifar-10, 1000 images of each class in test set
    clsses = 100
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


def train(num_classes, num_residual_units, k):

    workersocket = tcp_connection(FLAGS.ip_address, FLAGS.port)

    X, y_a, y_b, lam_tensor, parameters = initializer_34.create_placeholder()

    hp = ResNet_34.HParams(batch_size=FLAGS.mini_batch_size,
                        num_classes=num_classes,
                        num_residual_units=num_residual_units,
                        k=k,
                        initial_lr=FLAGS.lr)
    network = ResNet_34.ResNet18(hp=hp, images=X, labels_a=y_a, labels_b=y_b, lam=lam_tensor)
    network.build_model()
    # init = tf.global_variables_initializer()
    # trainable_vars = tf.trainable_variables()
    # update_vars_op = trainable_vars[0: 10]
    network.build_train_op()
    network.compute_acc()
    network.startup_bn()
    
    init = tf.global_variables_initializer()

    # trainable_vars = tf.trainable_variables()

    train_xs, train_ys, test_feeds = load_cifar100('./cifar-100-python/')

    saver = tf.train.Saver()

    replace = utils.replace_trainable_vars(network.trainable_vars, parameters)      
    # recv initial parameters from server

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

    with tf.device("/gpu: 0"):
        with tf.Session() as sess:
            seed = 1
            sess.run(init)
            num_mini_batch = int(train_xs.shape[0]/FLAGS.mini_batch_size)
            com_time_total = 0

            for epoch in range(1, FLAGS.nums_epoch+1):
                seed += 1
                epoch_cost = 0
                # avg_acc = 0
                com_time = 0
                mini_batches = utils.random_mini_batches(train_xs, train_ys, FLAGS.mini_batch_size, seed)
                worker_batches = utils.worker_random_minibatches(mini_batches, worker_minibatch_size=FLAGS.worker_batch_size,
                                                           seed=epoch + 1)
                for worker_batch in worker_batches:
                    (worker_batch_X, worker_batch_Y) = worker_batch[FLAGS.partition]
                    x, mix_x, target_a, target_b, lam = mixup_data(worker_batch_X, worker_batch_Y, alpha=1)
                    
                    replace_opt, cost, decay_lr, update_opt, train_acc, updated_paras = sess.run([replace, network.loss, network.lr,
                                                                                         network.update,
                                                                                         network.acc, network.update_vars_opt],
                             feed_dict={X: mix_x, y_a: target_a, y_b: target_b, lam_tensor: lam, parameters: p})

                    paras_dict = shape_process(updated_paras)
                    # print(len(pk.dumps(paras_dict)))
                    com_start = time.time()
                    new_parameters = push_parameters_to_server(workersocket, paras_dict)
                    com_end = time.time()
                    com_time += (com_end - com_start)
                    p = utils.convert_dict_to_tuple(new_parameters)
                    epoch_cost += cost / num_mini_batch
                costs.append(epoch_cost)
                com_time_total += com_time
                for test_feed in test_feeds:
                    test_acc, y_hat = sess.run([network.acc, network.logits], feed_dict={X: test_feed[0], y_a: test_feed[1]})
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
                # avg_acc = 0

                # print cost and train acc
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
            print(f"Total communication time for 300 epochs: {com_time_total:.4f} seconds")
        # print("Average test accuracy is {:.4f}".format(avg_acc))


def main():

    train(num_classes=100, num_residual_units=2, k=2)


if __name__=='__main__':

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




