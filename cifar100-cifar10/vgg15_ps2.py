import tensorflow as tf
import pickle as pk
import threading
from downsized_vgg15_init import initialize_parameters
import downsized_vgg15_functions
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('nums_worker', 1, """Number of workers.""")
tf.app.flags.DEFINE_integer('initial_parameters_length', 2375957, """Initial number of model parameters byte.""")
tf.app.flags.DEFINE_integer('expanded_parameters_length', 2048000000, """Expanded number of model parameters byte (for last 50 epochs).""")
tf.app.flags.DEFINE_integer('port', 2223, '''The port of parameter server''')
tf.app.flags.DEFINE_integer('total_epochs', 300, '''Total number of epochs''')


class BSP():

    def __init__(self, queue, init_parameters):
        self.nums_worker = FLAGS.nums_worker
        self.initial_parameters_length = FLAGS.initial_parameters_length
        self.expanded_parameters_length = 37781781
        self.queue = queue
        self.current_epoch = 1
        self.receive_count = 0
        self.lock = threading.Lock()
        self.aggregation_count = 1  # 聚合次数计数器
        self.parameters = init_parameters

    def recv(self, socket, worker_id):
        while True:
            data = b''
            ##################################################################### 1111
            if self.current_epoch == 2:
                self.receive_count += 1
                if self.receive_count >= 50:
            ##################################################################### 1111
                    self.current_epoch = 3
                    self.receive_count = 0  # Reset counter after updating epoch
            ##################################################################### 1111
            if self.current_epoch <= 2:
                while True:
                    pk_paras = socket.recv(2048000000)
                    if pk_paras == b'0x03':
                        self.queue['queue1'].put(b'0x03')
                        print(f'Received from worker {worker_id} finished!')
                        return
                    data += pk_paras
                    # print('aaa')
                    # print(len(data))
                    if len(data) == 2375957:
                        break

                paras_data = pk.loads(data)

                with self.lock:
                    self.current_epoch = paras_data['epoch']
                paras = paras_data['parameters']
                self.queue['queue1'].put(paras)


            else:
                while True:
                    pk_paras = socket.recv(2048000000)
                    if pk_paras == b'0x03':
                        self.queue['queue1'].put(b'0x03')
                        print(f'Received from worker {worker_id} finished!')
                        return
                    data += pk_paras
                    # print('bbb')
                    # print(len(data))
                    if len(data) == 18892513:
                        break
                paras_data = pk.loads(data)

                with self.lock:
                    self.current_epoch = paras_data['epoch']
                paras = paras_data['parameters']
                self.queue['queue1'].put(paras)
                # print(self.current_epoch)



    def predict(self):
        x = downsized_vgg15_functions.create_placeholder()
        parameters_tensor = downsized_vgg15_functions.inference(x)
        saver = tf.compat.v1.train.Saver()

        with tf.compat.v1.Session() as sess:
            saver.restore(sess, './train_stage_5304/model/model.ckpt')

            while True:
                # Fetch the current epoch safely
                with self.lock:
                    current_epoch = self.current_epoch

                ###################################################### 1111
                if current_epoch <= 2:
                    paras = np.zeros([1, 296970])  # Initialize parameter array

                    # Collect parameters from workers
                    for i in range(self.nums_worker):
                        conv_paras = self.queue["queue1"].get()
                        if conv_paras == b'0x03':
                            # Send finish signal to all workers
                            for k in range(self.nums_worker):
                                self.queue['queue' + str(k + 2)].put(b'0x03')
                            print('Aggregation finished!')
                            return
                        else:
                            paras += conv_paras / self.nums_worker

                    paras_pre = paras[0, 0:260160]
                    layer_params13_w = paras[0, 260160:285760]
                    layer_params13_b = paras[0, 285760:285860]
                    layer_params14_w = paras[0, 285860:295860]
                    layer_params14_b = paras[0, 295860:295960]
                    last_layer_params_w = paras[0, 295960:296960]
                    last_layer_params_b = paras[0, 296960:296970]

                    paras_pre = paras_pre.reshape([-1, 8, 8, 4065])
                    predicted_parameters = sess.run(parameters_tensor, feed_dict={x: paras_pre})

                    predicted_parameters['w13'] = np.reshape(layer_params13_w, [256, 100])
                    predicted_parameters['b13'] = np.reshape(layer_params13_b, [100])
                    predicted_parameters['w14'] = np.reshape(layer_params14_w, [100, 100])
                    predicted_parameters['b14'] = np.reshape(layer_params14_b, [100])
                    predicted_parameters['w15'] = np.reshape(last_layer_params_w, [100, 10])
                    predicted_parameters['b15'] = np.reshape(last_layer_params_b, [10])

                    for i in range(self.nums_worker):
                        self.queue['queue' + str(i + 2)].put(predicted_parameters)
                else:
                    xs = []
                    while True:
                        self.parameters = dict.fromkeys(self.parameters, 0)
                        for i in range(self.nums_worker):
                            paras = self.queue["queue1"].get()
                            # print(len(pk.dumps(paras)))
                            if paras == b'0x03':
                                for k in range(self.nums_worker):
                                    self.queue['queue' + str(k + 2)].put(b'0x03')
                                print('aggregation finished!')
                                return
                            else:
                                #####################################################
                                for j in range(15):  # 聚合来自各个worker的模型参数
                                    self.parameters["w" + str(j + 1)] += paras["w" + str(j + 1)] / self.nums_worker
                                    self.parameters["b" + str(j + 1)] += paras["b" + str(j + 1)] / self.nums_worker
                        # print(len(pk.dumps(self.parameters)))
                        # print(self.parameters)
                        for i in range(self.nums_worker):
                            self.queue['queue' + str(i + 2)].put(self.parameters)

    def send(self, socket, queue, worker_id):
        while True:
            parameters = queue.get()
            if parameters != b'0x03':
                socket.send(pk.dumps(parameters))
            else:
                print(f'Sent to worker {worker_id} finished!')
                socket.close()
                return


def main():
    init_parameters = initialize_parameters()
    server_socket = downsized_vgg15_functions.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    queue = downsized_vgg15_functions.create_queue(FLAGS.nums_worker)
    sockets = []
    worker_id = 1
    # Ensure that init_parameters is passed to BSP constructor
    bsp = BSP(queue, init_parameters)
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        if len(sockets) == FLAGS.nums_worker:
            for socket in sockets:
                bsp_init = threading.Thread(target=downsized_vgg15_functions.send_init_parameters,
                                            args=(socket, init_parameters, worker_id))
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
                bsp_send = threading.Thread(target=bsp.send,
                                            args=(socket, queue['queue' + str(worker_id + 1)], worker_id,))
                bsp_send.setDaemon(True)
                bsp_send.start()


if __name__ == '__main__':
    main()
