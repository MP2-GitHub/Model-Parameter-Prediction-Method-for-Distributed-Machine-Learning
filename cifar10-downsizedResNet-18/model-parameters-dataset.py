"""

code: model-parameters-dataset

processing model parameters to become a hierarchical model parameter dataset.

"""


import pickle as pk
import numpy as np
import time


def data_process():
    with open('inputs', 'rb') as f:

        data = pk.load(f)
        print(len(data))

    dataset = np.zeros([1, 260160])
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

        if k % 500 == 0:
            with open('./train_xs_{}'.format(k), 'wb') as f:
                f.write(pk.dumps(dataset[1:]))
                dataset = np.zeros([1, 260160])
                print("----------------Saved {} training samples successfully!---------------------".format(k))


def data_process2():
    with open('./labels', 'rb') as f:
        data = pk.load(f)

    # y1 = np.zeros([1, 73856])
    # y2 = np.zeros([1, 147584])
    y3 = np.zeros([1, 295168])
    y4 = np.zeros([1, 590080])
    y5 = np.zeros([1, 590080])
    y6 = np.zeros([1, 590080])
    y7 = np.zeros([1, 10250])
    xs1 = np.zeros([1, 1])
    k = 0
    t = 0
    j = 0
    start = time.time()
    for d in data:
        k += 1

        for i in d:
            j += 1
            temp = i.reshape(1, -1)
            xs1 = np.append(xs1, temp)

            if j == 2:
                y3 = np.r_[y3, xs1[1:].reshape(1, -1)]
                xs1 = np.zeros([1, 1])

            if j == 4:
                y4 = np.r_[y4, xs1[1:].reshape(1, -1)]
                xs1 = np.zeros([1, 1])

            if j == 6:
                y5 = np.r_[y5, xs1[1:].reshape(1, -1)]
                xs1 = np.zeros([1, 1])

            if j == 8:
                y6 = np.r_[y6, xs1[1:].reshape(1, -1)]
                xs1 = np.zeros([1, 1])

            if j == 10:
                y7 = np.r_[y7, xs1[1:].reshape(1, -1)]
                xs1 = np.zeros([1, 1])

            # if j == 12:
            #     y8 = np.r_[y8, xs1[1:].reshape(1, -1)]
            #     xs1 = np.zeros([1, 1])
            #
            # if j == 14:
            #     y9 = np.r_[y9, xs1[1:].reshape(1, -1)]
            #     xs1 = np.zeros([1, 1])

        j = 0
        xs1 = np.zeros([1, 1])

        if k % 100 == 0:
            end = time.time()
            speed = (100 / (end - start))
            print('标签预处理：这是第{}个标签样本，当前标签的维度：y3 = {}，y4 = {}, y5 = {}, y6 = {}, y7 = {}, '
                  '文件保存：{}，处理速度：{:.2f} samples/s'
                  .format(k, y3.shape, y4.shape, y5.shape, y6.shape, y7.shape, t, speed))
            start = time.time()
        if k % 500 == 0:
            # with open('./dataset/label-1_{}'.format(k), 'wb') as f:
            #     f.write(pk.dumps(y1[1:]))
            #     y1 = np.zeros([1, 73856])
            # with open('./dataset/label-2_{}'.format(k), 'wb') as f:
            #     f.write(pk.dumps(y2[1:]))
            #     y2 = np.zeros([1, 147584])
            with open('label-3_{}'.format(k), 'wb') as f:
                f.write(pk.dumps(y3[1:]))
                y3 = np.zeros([1, 295168])

            with open('label-4_{}'.format(k), 'wb') as f:
                f.write(pk.dumps(y4[1:]))
                y4 = np.zeros([1, 590080])

            with open('label-5_{}'.format(k), 'wb') as f:
                f.write(pk.dumps(y5[1:]))
                y5 = np.zeros([1, 590080])

            with open('label-6_{}'.format(k), 'wb') as f:
                f.write(pk.dumps(y6[1:]))
                y6 = np.zeros([1, 590080])

            with open('label-7_{}'.format(k), 'wb') as f:
                f.write(pk.dumps(y7[1:]))
                y7 = np.zeros([1, 10250])

                t += 1
                print('------------------已保存 {} 个标签样本的整型文件-------------------'.format(k))


def read_dataset():

    index = list(np.arange(250, 4750, 250))
    # y1 = np.zeros([1, 73856])
    # y2 = np.zeros([1, 147584])
    y5 = np.zeros([1, 295168])
    y6 = np.zeros([1, 590080])
    y7 = np.zeros([1, 590080])
    y8= np.zeros([1, 590080])
    y9 = np.zeros([1, 10250])
    for k in index:

        # with open('./dataset/label-5_{}'.format(k), 'rb') as f:
        #     data5 = pk.load(f)
        #     y5 = np.r_[y5, data5]
        # with open('./dataset/label-6_{}'.format(k), 'rb') as f:
        #     data6 = pk.load(f)
        #     y6 = np.r_[y6, data6]
        # with open('./dataset/label-7_{}'.format(k), 'rb') as f:
        #     data7 = pk.load(f)
        #     y7 = np.r_[y7, data7]
        # with open('./dataset/label-8_{}'.format(k), 'rb') as f:
        #     data8 = pk.load(f)
        #     y8 = np.r_[y8, data8]
        with open('./dataset/label-9_{}'.format(k), 'rb') as f:
            data9 = pk.load(f)
            y9 = np.r_[y9, data9]

    # with open('./dataset/y1', 'wb') as f:
    #     f.write(pk.dumps(y1[1:], protocol=4))
    #
    # with open('./dataset/y2', 'wb') as f:
    #     f.write(pk.dumps(y2[1:], protocol=4))

    # with open('./dataset/y5', 'wb') as f:
    #     f.write(pk.dumps(y5[1:], protocol=4))

    # with open('./dataset/y6', 'wb') as f:
    #     f.write(pk.dumps(y6[1:], protocol=4))
    #
    # with open('./dataset/y7', 'wb') as f:
    #     f.write(pk.dumps(y7[1:], protocol=4))
    #
    # with open('./dataset/y8', 'wb') as f:
    #     f.write(pk.dumps(y8[1:], protocol=4))

    with open('./dataset/y9', 'wb') as f:
        f.write(pk.dumps(y9[1:], protocol=4))

    # print(y5[1:].shape)
    # print(y6[1:].shape)
    # print(y7[1:].shape)
    # print(y8[1:].shape)
    print(y9[1:].shape)
    print('Finished!')


def model_parameters_dataset(batch_size):
    with open('./dataset/train_xs', 'rb') as f:
        input = pk.load(f)

    with open('./dataset/y1', 'rb') as f:
        y1 = pk.load(f)
    with open('./dataset/y2', 'rb') as f:
        y2 = pk.load(f)
    with open('./dataset/y3', 'rb') as f:
        y3 = pk.load(f)
    with open('./dataset/y4', 'rb') as f:
        y4 = pk.load(f)
    with open('./dataset/y5', 'rb') as f:
        y5 = pk.load(f)
    with open('./dataset/y6', 'rb') as f:
        y6 = pk.load(f)
    with open('./dataset/y7', 'rb') as f:
        y7 = pk.load(f)

    xs = []
    m = input.shape[0]
    nums_batches = m // batch_size
    for k in range(0, nums_batches):
        train_xs = input[batch_size * k: batch_size * (k + 1)]
        ys1 = y1[batch_size * k: batch_size * (k + 1)]
        ys2 = y2[batch_size * k: batch_size * (k + 1)]
        ys3 = y3[batch_size * k: batch_size * (k + 1)]
        ys4 = y4[batch_size * k: batch_size * (k + 1)]
        ys5 = y5[batch_size * k: batch_size * (k + 1)]
        ys6 = y6[batch_size * k: batch_size * (k + 1)]
        ys7 = y7[batch_size * k: batch_size * (k + 1)]

        xs.append((train_xs, ys1, ys2, ys3, ys4, ys5, ys6, ys7))

    return xs, nums_batches


def read_dataset2():

    K = list(np.arange(start=500, stop=5000, step=500))
    D = np.zeros([1, 38720])

    for k in K:
        with open('./train_xs12_{}'.format(k), 'rb') as f:
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
    with open('dataset/xs_12_vgg13', 'wb') as f:
        f.write(pk.dumps(D[1:], protocol=4))
        print('已经保存数据集！')
        print('数据集维度：{}'.format(D[1:].shape))


def add_layers():

    with open('./dataset/xs_layer12345', 'rb') as f:
        input = pk.load(f)

    with open('./dataset/y6', 'rb') as f:
        y6 = pk.load(f)

    m = input.shape[0]
    newdata = np.zeros([1, 1145408])
    k = 0
    for i in range(m):
        k += 1
        layers_12345 = input[i]
        layers_6 = y6[i]
        layers_12345 = layers_12345.reshape(1, 555328)
        layers_6 = layers_6.reshape(1, 590080)
        data = np.append(layers_12345, layers_6, axis=1)
        newdata = np.append(newdata, data, axis=0)

        if k % 100==0:
            print('Current dataset shape is {}'.format(newdata[1:].shape))

        if k % 500 == 0:
            with open('./xs_layer123456_{}'.format(k), 'wb') as f:
                f.write(pk.dumps(newdata[1:], protocol=4))
                newdata = np.zeros([1, 1145408])
                print('已经保存数据集{}！'.format(k))

    l = list(np.arange(500, 5000, 500))
    for i in l:
        with open('./xs_layer123456_{}'.format(i), 'rb') as f:
            d = pk.load(f)
        newdata = np.append(newdata, d, axis=0)

    newdata = newdata[1:]
    print(newdata.shape)
    with open('dataset/xs_layer123456', 'wb') as f:
        f.write(pk.dumps(newdata))

    print('----------------------Finished!-------------------------')


def model_parameters_dataset2(batch_size):

    xs1 = []
    xs2 = []
    xs3 = []
    xs4 = []
    xs5 = []
    xs6 = []
    xs7 = []
    xs8 = []
    xs9 = []
    nums_batches = int(4500/batch_size)
    L = list(np.arange(start=500, stop=5000, step=500))
    for i in L:
        with open('dataset/xs_layers_1234567_{}'.format(i), 'rb') as f:
            data = pk.load(f)
            train_data = data
            for k in range(0, nums_batches):
                train_xs = train_data[batch_size * k: batch_size * (k + 1)]
                if i == 500:
                    xs1.append(train_xs)
                elif i == 1000:
                    xs2.append(train_xs)
                elif i == 1500:
                    xs3.append(train_xs)
                elif i == 2000:
                    xs4.append(train_xs)
                elif i == 2500:
                    xs5.append(train_xs)
                elif i == 3000:
                    xs6.append(train_xs)
                elif i == 3500:
                    xs7.append(train_xs)
                elif i == 4000:
                    xs8.append(train_xs)
                elif i == 4500:
                    xs9.append(train_xs)
            if i % 500 == 0:
                print("Completed {} iterations!".format(i))

    print("Complete dataset loading! The shape of dataset is {}. Start the \"Predict Network\" training!"
          .format(nums_batches))

    return xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, nums_batches


def extra_layers():

    K = list(np.arange(start=250, stop=4750, step=250))
    D = np.zeros([1, 555328])
    t = 0
    for k in K:
        with open('./parameters_{}'.format(k), 'rb') as f:
            data = pk.load(f)

            for paras_1234567 in data:
                t += 1
                train_xs = paras_1234567[0, 0:555328]
                D = np.r_[D, train_xs.reshape(1, 555328)]

                if t % 100 == 0:
                    print('----------------completed {} iterations combine!-----------------'.format(t))

                if t % 500 == 0:
                    with open('./xs_layers_12345_{}'.format(t), 'wb') as f:
                        f.write(pk.dumps(D[1:], protocol=4))
                        print('已经保存数据集！')
                        print('数据集维度：{}'.format(D[1:].shape))
                    D = np.zeros([1, 555328])

extra_layers()





