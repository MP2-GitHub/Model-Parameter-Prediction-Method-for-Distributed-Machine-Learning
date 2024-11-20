import tensorflow as tf
import pickle as pk
import threading
from initializer_34 import initial_parameters
import utils
import numpy as np
import random

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_integer('nums_worker', 1, """Number of workers.""")
tf.app.flags.DEFINE_integer('parameters_length', 22762575, """Number of gradients byte.""")

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')

class BSP():

    def __init__(self, init_parameters, queue, len_of_parameters):
        self.nums_worker = FLAGS.nums_worker
        self.L = FLAGS.parameters_length
        self.parameters = init_parameters
        self.queue = queue
        self.l = len_of_parameters
        self.t = 0
        self.ep = 0
        self.nums_mini_batches = 50
        self.k = 0
        self.iterations = list(np.arange(start=50 * self.ep, stop=50 * self.ep + 50, step=1))

    def recv(self, socket, worker_id):
        while True:
            data = b''
            while True:
                pk_paras = socket.recv(2048000000)
                if pk_paras == b'0x03':
                    self.queue['queue1'].put(b'0x03')
                    print('recv {} finished!'.format(worker_id))
                    return  # 结束线程
                data += pk_paras
                if len(data) == self.L:
                    break
            paras = pk.loads(data)
            # print(len(paras))
            self.queue['queue1'].put(paras)

    def send(self, socket, queue, worker_id):

        while True:
            parameters = queue.get()
            if parameters != b'0x03':
                socket.send(pk.dumps(parameters))
            else:
                print('send {} finished!'.format(worker_id))
                socket.close()
                return

    def aggregation(self):
        xs = []
        while True:
            self.parameters = dict.fromkeys(self.parameters, 0)
            for i in range(self.nums_worker):
                paras = self.queue["queue1"].get()
                # print(len(pk.dumps(paras)))
                if paras == b'0x03':
                    for k in range(self.nums_worker):
                        self.queue['queue'+str(k+2)].put(b'0x03')
                    print('aggregation finished!')
                    return
                else:
                    for j in range(self.l):  # 聚合来自各个worker的模型参数

                        self.parameters["w" + str(j + 1)] += paras["w" + str(j + 1)] / self.nums_worker
                        self.parameters["b" + str(j + 1)] += paras["b" + str(j + 1)] / self.nums_worker
            # print(len(pk.dumps(self.parameters)))
            for i in range(self.nums_worker):
                self.queue['queue' + str(i+2)].put(self.parameters)
            # 模型参数采样
            random.seed(self.ep)
            random_list = random.sample(self.iterations, 15)
            if self.t in random_list:
               xs.append(self.save_layers())
               if len(xs) % 150 == 0:
                   self.k += 1
                   with open('./resnet34_parameters_{}'.format(self.k*150), 'wb') as f:
                       f.write(pk.dumps(xs, protocol=4))
                       print('------------Save success!-----------------')
                       print('Current shape is ({}, {})'.format(len(xs), xs[-1].shape[1]))
                   xs = []
            # 计数
            self.counter()

    def save_layers(self):

        conv_parameters = np.zeros([1, 1])
        for i in range(self.l):
            if i < 17:
                w = self.parameters['w' + str(i + 1)]
                b = self.parameters['b' + str(i + 1)]
                conv_parameters = np.append(conv_parameters, w.reshape([1, -1]))
                conv_parameters = np.append(conv_parameters, b.reshape([1, -1]))
            else:
                break
        conv_parameters = conv_parameters[1:].reshape([1, -1])
        return conv_parameters

    def counter(self):
        self.t += 1
        if self.t % self.nums_mini_batches == 0:      # 1个epoch
            self.ep += 1
            self.iterations = list(np.arange(start=50 * self.ep, stop=50 * self.ep + 50, step=1))
            print('BSP: Epoch = {}/{}, Saved files: x{}'.format(self.ep, 300, self.k))

def main():

    init_parameters = initial_parameters()
    len_of_parameters = len(init_parameters)//2  # 8
    server_socket = utils.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    queue = utils.create_queue(FLAGS.nums_worker)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        if len(sockets) == FLAGS.nums_worker:
            # 实例化
            bsp = BSP(init_parameters,
                      queue,
                      len_of_parameters)
            for socket in sockets:

                bsp_init = threading.Thread(target=utils.send_init_parameters, args=(socket, init_parameters, worker_id))
                bsp_init.start()
                bsp_init.join()
                worker_id += 1

            worker_id = 1
            for socket in sockets:

                bsp_recv = threading.Thread(target=bsp.recv, args=(socket, worker_id,))
                bsp_recv.setDaemon(True)
                bsp_recv.start()
                worker_id += 1

            bsp_agg = threading.Thread(target=bsp.aggregation, args=())
            bsp_agg.setDaemon(True)
            bsp_agg.start()

            worker_id = 1
            for socket in sockets:
                bsp_send = threading.Thread(target=bsp.send, args=(socket, queue['queue' + str(worker_id + 1)], worker_id,))
                bsp_send.setDaemon(True)
                bsp_send.start()
                worker_id += 1

if __name__=='__main__':
    main()
