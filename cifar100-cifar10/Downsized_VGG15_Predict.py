import tensorflow as tf
import pickle as pk
import threading
from downsized_vgg15_init import initialize_parameters
import downsized_vgg15_functions
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('nums_worker', 1, """Number of workers.""")
tf.app.flags.DEFINE_integer('parameters_length', 2375921, """Number of model parameters byte.""")
tf.app.flags.DEFINE_integer('port', 2227, '''The port of parameter server''')


class BSP():

    def __init__(self, queue):
        self.nums_worker = FLAGS.nums_worker
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
                    return
                data += pk_paras
                if len(data) == self.L:
                    break
            paras = pk.loads(data)
            self.queue['queue1'].put(paras)

    def predict(self):
        x = downsized_vgg15_functions.create_placeholder()
        parameters_tensor = downsized_vgg15_functions.inference(x)
        saver = tf.train.Saver()  # 创建 Saver 对象

        with tf.Session() as sess:
            saver.restore(sess, './train_stage_5304/model/model.ckpt')  # 恢复模型参数
            while True:
                paras = np.zeros([1, 296970])  # 接受参数
                layer_params13_w = np.zeros([1, 256 * 100])  # 用于存储 w13
                layer_params13_b = np.zeros([1, 100])  # 用于存储 b13
                layer_params14_w = np.zeros([1, 100 * 100])  # 用于存储 w14
                layer_params14_b = np.zeros([1, 100])  # 用于存储 b14
                last_layer_params_w = np.zeros([1, 100 * 10])  # 用于存储 w15
                last_layer_params_b = np.zeros([1, 10])  # 用于存储 b15

                # 从 worker 收集参数
                for i in range(self.nums_worker):
                    conv_paras = self.queue["queue1"].get()  # 从队列中获取参数
                    if conv_paras == b'0x03':  # 如果是结束信号
                        for k in range(self.nums_worker):
                            self.queue['queue' + str(k + 2)].put(b'0x03')  # 发送结束信号给所有 worker
                        print('aggregation finished!')
                        return
                    else:
                        paras += conv_paras / self.nums_worker
                    # print(conv_paras.shape)
                    # 将前10层参数累加

                paras_pre = paras[0, 0:260160]


                layer_params13_w = paras[0, 260160:285760]
                layer_params13_b = paras[0, 285760:285860]
                layer_params14_w = paras[0, 285860:295860]
                layer_params14_b = paras[0, 295860:295960]
                last_layer_params_w = paras[0, 295960:296960]
                last_layer_params_b = paras[0, 296960:296970]



                # 预测其余参数（后11层）
                paras_pre = paras_pre.reshape([-1, 8, 8, 4065])
                predicted_parameters = sess.run(parameters_tensor, feed_dict={x: paras_pre})

                # 替换预测中的输出参数
                predicted_parameters['w13'] = np.reshape(layer_params13_w, [256, 100])
                predicted_parameters['b13'] = np.reshape(layer_params13_b, [100])
                predicted_parameters['w14'] = np.reshape(layer_params14_w, [100, 100])
                predicted_parameters['b14'] = np.reshape(layer_params14_b, [100])
                predicted_parameters['w15'] = np.reshape(last_layer_params_w, [100, 10])
                predicted_parameters['b15'] = np.reshape(last_layer_params_b, [10])

                # 将更新后的参数发送回给所有 worker
                for i in range(self.nums_worker):
                    self.queue['queue' + str(i + 2)].put(predicted_parameters)


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
    init_parameters = initialize_parameters()
    server_socket = downsized_vgg15_functions.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    queue = downsized_vgg15_functions.create_queue(FLAGS.nums_worker)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        if len(sockets) == FLAGS.nums_worker:
            bsp = BSP(queue)
            for socket in sockets:
                bsp_init = threading.Thread(target=downsized_vgg15_functions.send_init_parameters, args=(socket, init_parameters, worker_id))
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

if __name__ == '__main__':
    main()
