import pickle as pk
import numpy as np


# x=(m, 2572) <---> y1=(m, 48120), y2=(m, 10164), y3=(m, 850)

def data_process():
    with open('save_parameters2', 'rb') as f:

        data = pk.load(f)

    dataset = np.zeros([1, 2572])
    xs = np.zeros([1, 1])
    k = 0
    for d in data:
        k += 1
        for i in d:
            xs = np.append(xs, i)
        dataset = np.r_[dataset, xs[1:].reshape(1, -1)]
        xs = np.zeros([1, 1])
        if k % 100 == 0:
            print('{}次迭代后，数据集的维度：{}'.format(k, dataset.shape))

    with open('parameters_dataset', 'wb') as f:
        f.write(pk.dumps(dataset))


def read_dataset():

    with open('parameters_dataset', 'rb') as f:

        data = pk.load(f)

    train_xs = data[1:]
    print(train_xs)


def extra_layers():

    K = list(np.arange(start=150, stop=4650, step=150))
    D = np.zeros([1, 2572])
    t = 0
    for k in K:
        with open('./paras_dataset/lenet5_parameters_{}'.format(k), 'rb') as f:
            data = pk.load(f)

            for paras in data:
                t += 1
                train_xs = paras[0, 0:2572]
                D = np.r_[D, train_xs.reshape(1, 2572)]

                if t % 100 == 0:
                    print('----------------completed {} iterations combine!-----------------'.format(t))

                if t % 500 == 0:
                    with open('./paras_dataset/lenet5_layer12_{}'.format(t), 'wb') as f:
                        f.write(pk.dumps(D[1:], protocol=4))
                        print('已经保存数据集！')
                        print('数据集维度：{}'.format(D[1:].shape))
                    D = np.zeros([1, 2572])


def read_dataset2():

    K = list(np.arange(start=500, stop=5000, step=500))
    D = np.zeros([1, 2572])

    for k in K:
        with open('./paras_dataset/lenet5_layer12_{}'.format(k), 'rb') as f:
            data = pk.load(f)
            train_xs = data
            D = np.r_[D, train_xs]
        print('已完成 {} 个样本合并'.format(k))
        #if k % 500==0:
        #    with open('dataset/xs_layers_123456_{}'.format(k), 'wb') as f:
        #        f.write(pk.dumps(D[1:], protocol=4))
        #        print('已经保存数据集！')
        #        print('数据集维度：{}'.format(D[1:].shape))
        #    D = np.zeros([1, 1145408])
    with open('./paras_dataset/xs_12_lenet5', 'wb') as f:
        f.write(pk.dumps(D[1:], protocol=4))
        print('已经保存数据集！')
        print('数据集维度：{}'.format(D[1:].shape))
