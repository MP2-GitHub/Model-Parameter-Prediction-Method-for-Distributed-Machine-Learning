import numpy as np
from input_data import Cifar10
import pickle as pk
import math
import random
import matplotlib.pyplot as plt
import matplotlib
font = {'family':'Microsoft YaHei', 'weight':'bold', 'size':12}
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', **font)


def compute_recall(y_hat, y):

    n = y.shape[0]
    record = {}               # 正确判为正类
    count = {}                # 统计所有
    nums_each_class = 2    # cifar-10, 1000 images of each class in test set
    clsses = 4
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

    return record, recalls, avg_recall, count, precisions, avg_pred, F1

# test
# y = np.array([[0, 0, 0, 1],   # 3
#               [1, 0, 0, 0],   # 0
#               [0, 0, 0, 1],   # 3
#               [0, 0, 1, 0],   # 2
#               [1, 0, 0, 0],   # 0
#               [0, 1, 0, 0],   # 1
#               [0, 1, 0, 0],   # 1
#               [0, 0, 1, 0]])  # 2
#
# y_hat = np.array([[0.02, 0.03, 0.05, 0.9], # 3
#                   [0.1, 0.3, 0.5, 0.1],    # 2
#                   [0, 0.01, 0.01, 0.98],   # 3
#                   [0.05, 0.09, 0.90, 0.05], # 2
#                   [0.88, 0.1, 0.01, 0.01],  # 0
#                   [0.89, 0.01, 0.05, 0.05], # 0
#                   [0.05, 0.9, 0.04, 0.01],  # 1
#                   [0.01, 0.04, 0.9, 0.05]]) # 2
#
#
# record, recalls, avg_recall, count, precisions, avg_pred, F1 = compute_recall(y_hat=y_hat, y=y)
#
# print('Recalls:', recalls)
# print('Avg Recall:', avg_recall)
# print('Count:', count)
# print('Precisions:', precisions)
# print('Avg precision:', avg_pred)
# print('F1 score:', F1)


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


def images_sort(dataset, label):

    n = label.shape[0]
    print(n)
    classes = {}
    temp = []
    count = 0
    for i in range(n):
        count += 1
        image = dataset[i]
        c = np.argmax(label[i])
        if c not in classes:
            classes[c] = image.reshape(1, 32, 32, 3)
        else:
            classes[c] = np.append(classes[c], image.reshape(1, 32, 32, 3), axis=0)
        if count % 1000 == 0:
            print('当前是第{}张图像，目前{}类的维度是{}'.format(count,  c+1, classes[c].shape))
        if count % 10000 == 0:
            temp.append(classes.copy())
            classes.clear()
    classes = {}
    for cs in temp:
        for k, v in cs.items():
            if k not in classes:
                classes[k] = v
            else:
                classes[k] = np.append(classes[k], v, axis=0)
    print('------------------------------train data-------------------------------------')
    for k, v in classes.items():
        print('This is the {} class, its shape is {}'.format(k+1, v.shape))

    # one-hot labels
    labes_one_hot = {}
    num_classes = 10
    n = classes[0].shape[0]
    print('-------------------------------train labels-------------------------------------')
    print('Each label has {} images.'.format(n))
    for i in range(num_classes):
        labes_one_hot[i] = np.zeros([n, num_classes])

    for k in classes.keys():
        for i in range(n):
            labes_one_hot[k][i, k] = 1

    print('Completed the one-hot label:')
    print(labes_one_hot)

    data = {}
    data['train_data'] = classes
    data['labels'] = labes_one_hot

    with open('./cifar-10-classed', 'wb') as f:
        f.write(pk.dumps(data, protocol=4))

    return classes


# train_xs, train_labels, test_xs, test_labels = load_dataset()
#
# classes = images_sort(train_xs, train_labels)


def load_data(path):

    with open(path, 'rb') as f:
        dataset = pk.load(f)
    return dataset


def sample_image(dataset, num_classes, percent, seed, mini_batch_size):

    train_data = dataset['train_data']
    labels = dataset['labels']

    n = train_data[0].shape[0]
    nums_sample = int(n*percent)
    # print('Sample {} image from each class.'.format(nums_sample))
    xs = np.zeros([1, 32, 32, 3])
    ys = np.zeros([1, num_classes])

    np.random.seed(seed)
    l = list(np.arange(start=0, stop=n, step=1))
    random_index = random.sample(l, nums_sample)
    random_index.sort()
    for k, v in train_data.items():
        # print('{}: '.format(k), v[random_index, :, :, :].shape)
        xs = np.append(xs, v[random_index, :, :, :], axis=0)
        ys = np.append(ys, labels[k][random_index, :], axis=0)

    xs = xs[1:]
    ys = ys[1:]

    # print(xs.shape)
    # print(ys.shape)

    mini_batches = random_mini_batches(xs, ys, mini_batch_size=mini_batch_size, seed=seed)

    return mini_batches, len(mini_batches)

# test
# dataset = load_data(path='./cifar-10-classed',)
# sample_image(dataset, 10, 0.1, 0, 125)


def show():       # 采样朝参数

    # x1 = np.arange(5, 45, 5)
    x1 = ['5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%']  # 图像采样率
    x2 = np.arange(1, 8, 1)

    y1 = [0.8029, 0.8075, 0.8079, 0.8129, 0.8154, 0.8110, 0.8122, 0.8112, 0.8136, 0.8135]   # 训练预测模型的精度
    y2 = [10, 20, 31, 41, 51, 61, 72, 83, 94, 104]    # 批量处理时间
    y3 = [0.7427, 0.7773, 0.7929, 0.8075, 0.8166, 0.8306, 0.8419] # 训练预测模型的精度
    # y4 = [0.007, 0.148, 0.42, 0.99, 2.12, 4.37]
    y4 = ['0.077%', '1.66%', '4.82%', '11.1%', '23.8%', '49%', '74%']  # 前层的模型参数数量，通信量

    fig = plt.figure(figsize=(8, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    # h1, = ax.plot(x2, y3, color='b', linewidth=2, label='Accuracy', ls='--', marker='*')
    h1, = ax.plot(x1, y1, color='b', linewidth=2, label='Accuracy', ls='--', marker='*', ms=8)
    ax.set_ylabel('Accuracy')
    # ax.set_title("Accuracy and Training Time")
    ax.set_xlabel('Image Sample Rate')
    # ax.set_xlabel('Layers accumulate')

    ax2 = ax.twinx()
    h2, = ax2.plot(x1, y2, color='r', linewidth=2,
                   label='Traffic', ls='--', marker='D', ms=8)
    # ax2.set_ylabel('Proportion of model parameters')
    ax2.set_ylabel('Batch training time (sec/batch)')

    ax.spines['left'].set_color('blue')
    ax2.spines['right'].set_color('r')
    ax.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='r')
    ax.yaxis.label.set_color('blue')
    ax2.yaxis.label.set_color('r')

    ax.legend([h1, h2], ['Accuracy', 'Batch training time'], loc='upper left')
    # plt.legend()
    plt.show()

