import pickle as pk
import numpy as np


l = list(np.arange(250, stop=4750, step=250))

# xs = []
xs1234 = np.zeros([1, 260160])
xs = np.zeros([1, 1])
y5 = np.zeros([1, 295168])
y6 = np.zeros([1, 590080])
y7 = np.zeros([1, 590080])
y8 = np.zeros([1, 590080])
y9 = np.zeros([1, 10250])
total_xs = []
total_ys = []
L = 9
t = 0
K = 0
for i in l:
    with open('./parameters_{}'.format(str(i)), 'rb') as f:
        paras_list = pk.load(f)
        for paras in paras_list:
            t += 1
            for k in range(L):
                if k < 4:
                    xs_w = paras['w' + str(k + 1)].reshape(1, -1)
                    xs_b = paras['b' + str(k + 1)].reshape(1, -1)
                    xs = np.append(xs, xs_w)
                    xs = np.append(xs, xs_b)
            xs1234 = np.r_[xs1234, xs[1:].reshape(1, -1)]
            xs = np.zeros([1, 1])

            if t % 100==0:
                print("Combined {} iterations model parameters.".format(t))
            if t % 500 == 0:
                with open('./train_xs1234_{}'.format(t), 'wb') as f:
                    f.write(pk.dumps(xs1234[1:]))
                    print("Current shape is {}".format(xs1234[1:].shape))
                    xs1234 = np.zeros([1, 260160])
                    print("----------------Saved {} training samples successfully!---------------------".format(t))

        #         if k == 4:
        #             y5_w = paras['w'+str(k+1)].reshape(1, -1)
        #             y5_b = paras['b'+str(k+1)].reshape(1, -1)
        #             y5s = np.append(y5_w, y5_b)
        #         elif k == 5:
        #             y6_w = paras['w'+str(k+1)].reshape(1, -1)
        #             y6_b = paras['b'+str(k+1)].reshape(1, -1)
        #             y6s = np.append(y6_w, y6_b)
        #         elif k == 6:
        #             y7_w = paras['w'+str(k+1)].reshape(1, -1)
        #             y7_b = paras['b'+str(k+1)].reshape(1, -1)
        #             y7s = np.append(y7_w, y7_b)
        #         elif k == 7:
        #             y8_w = paras['w'+str(k+1)].reshape(1, -1)
        #             y8_b = paras['b'+str(k+1)].reshape(1, -1)
        #             y8s = np.append(y8_w, y8_b)
        #         elif k == 8:
        #             y9_w = paras['w'+str(k+1)].reshape(1, -1)
        #             y9_b = paras['b'+str(k+1)].reshape(1, -1)
        #             y9s = np.append(y9_w, y9_b)
        #
        #     y5 = np.r_[y5, y5s.reshape(1, -1)]
        #     y6 = np.r_[y6, y6s.reshape(1, -1)]
        #     y7 = np.r_[y7, y7s.reshape(1, -1)]
        #     y8 = np.r_[y8, y8s.reshape(1, -1)]
        #     y9 = np.r_[y9, y9s.reshape(1, -1)]
        #
        # if t % 250 == 0:
        #     K += 1
        #     print('标签预处理：这是第{}个标签样本，当前标签的维度：y5 = {}，y6 = {}, y7 = {}, y8 = {}, y9 = {}, 文件保存：{}'
        #           .format(t, y5[1:].shape, y6[1:].shape, y7[1:].shape, y8[1:].shape, y9[1:].shape, K))
        #
        #     with open('label-5_{}'.format(t), 'wb') as f:
        #         f.write(pk.dumps(y5[1:]))
        #         y5 = np.zeros([1, 295168])
        #
        #     with open('label-6_{}'.format(t), 'wb') as f:
        #         f.write(pk.dumps(y6[1:]))
        #         y6 = np.zeros([1, 590080])
        #
        #     with open('label-7_{}'.format(t), 'wb') as f:
        #         f.write(pk.dumps(y7[1:]))
        #         y7 = np.zeros([1, 590080])
        #
        #     with open('label-8_{}'.format(t), 'wb') as f:
        #         f.write(pk.dumps(y8[1:]))
        #         y8 = np.zeros([1, 590080])
        #
        #     with open('label-9_{}'.format(t), 'wb') as f:
        #         f.write(pk.dumps(y9[1:]))
        #         y9 = np.zeros([1, 10250])









