"""
code: mp2-ps
model: downsized-resnet-50
dataset: cifar-100
"""

import tensorflow as tf
import pickle as pk
import threading
from initializer_50 import initial_parameters
import utils
import ResNet50_functions
import numpy as np

FLAGS = tf.app.flags.FLAGS
# Neural Network Configuration
tf.app.flags.DEFINE_integer('nums_worker', 2, """Number of workers.""")
tf.app.flags.DEFINE_integer('parameters_length', 2744993, """Number of model parameters byte.""")

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 7777, '''The port of parameter server''')


class BSP():

    def __init__(self, queue):

        self.nums_worker = FLAGS.nums_worker
        # self.nums_mini_batch = FLAGS.nums_mini_batch
        self.L = FLAGS.parameters_length
        self.parameters = 0
        self.queue = queue

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
            self.queue['queue1'].put(paras)

    def predict(self):
        x = ResNet50_functions.create_placeholder()
        parameters_tensor = ResNet50_functions.inference(x)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, './save-model/model.ckpt')
            while True:
                paras = np.zeros([1, 343104])
                for i in range(self.nums_worker):
                    conv_paras = self.queue["queue1"].get()
                    if conv_paras == b'0x03':
                        for k in range(self.nums_worker):
                            self.queue['queue' + str(k + 2)].put(b'0x03')
                        print('aggregation finished!')
                        return
                    else:
                        paras += conv_paras / self.nums_worker
                paras = paras.reshape([-1, 8, 8, 5361])
                self.parameters = sess.run(parameters_tensor, feed_dict={x: paras})
                # print(len(pk.dumps(self.parameters)))
                for i in range(self.nums_worker):
                    self.queue['queue' + str(i + 2)].put(self.parameters)

    def send(self, socket, queue, worker_id):

        while True:
            parameters = queue.get()
            if parameters != b'0x03':
                socket.send(pk.dumps(parameters))
            else:
                print('send {} finished!'.format(worker_id))
                socket.close()
                return

def main():

    init_parameters = initial_parameters()
    server_socket = ResNet50_functions.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    queue = ResNet50_functions.create_queue(FLAGS.nums_worker)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        if len(sockets) == FLAGS.nums_worker:
            # 实例化
            bsp = BSP(queue)
            for socket in sockets:
                bsp_init = threading.Thread(target=ResNet50_functions.send_init_parameters, args=(socket, init_parameters, worker_id))
                bsp_init.start()
                bsp_init.join()
                worker_id += 1
            worker_id = 1
            for socket in sockets:
                bsp_recv = threading.Thread(target=bsp.recv, args=(socket, worker_id,))
                bsp_recv.setDaemon(True)
                bsp_recv.start()
                worker_id += 1

            bsp_agg = threading.Thread(target=bsp.predict, args=())
            bsp_agg.setDaemon(True)
            bsp_agg.start()

            worker_id = 1
            for socket in sockets:

                bsp_send = threading.Thread(target=bsp.send, args=(socket, queue['queue'+str(worker_id+1)], worker_id,))
                bsp_send.setDaemon(True)
                bsp_send.start()
                worker_id += 1


if __name__=='__main__':
    main()

