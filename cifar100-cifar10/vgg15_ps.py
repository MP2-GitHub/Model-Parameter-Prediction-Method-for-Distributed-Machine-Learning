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

    def __init__(self, queue):
        self.nums_worker = FLAGS.nums_worker
        self.initial_parameters_length = FLAGS.initial_parameters_length
        self.expanded_parameters_length = 37781781
        self.queue = queue
        self.current_epoch = 1
        self.receive_count = 0
        self.lock = threading.Lock()
        self.aggregation_count = 1  # 聚合次数计数器

    def recv(self, socket, worker_id):
        while True:
            data = b''
            ##################################################################### 1111
            if self.current_epoch == 250:
                self.receive_count += 1
                if self.receive_count >= 50:
            ##################################################################### 1111
                    self.current_epoch = 251
                    self.receive_count = 0  # Reset counter after updating epoch
            ##################################################################### 1111
            if self.current_epoch <= 250:
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
                    if len(data) == 37781781:
                        break
                paras_data = pk.loads(data)

                with self.lock:
                    self.current_epoch = paras_data['epoch']
                paras = paras_data['parameters']
                self.queue['queue1'].put(paras)
                print(self.current_epoch)



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
                if current_epoch <= 250:
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
                    paras = np.zeros([1, 4722698])  # Initialize parameter array for expanded parameters

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

                    layer_params1_w = paras[0, 0:1728]
                    layer_params1_b = paras[0, 1728: 1792]
                    layer_params2_w = paras[0, 1792: 38656]
                    layer_params2_b = paras[0, 38656: 38720]
                    layer_params3_w = paras[0, 38720: 112448]
                    layer_params3_b = paras[0, 112448: 112576]
                    layer_params4_w = paras[0, 112576: 260032]
                    layer_params4_b = paras[0, 260032: 260160]
                    layer_params5_w = paras[0, 260160: 555072]
                    layer_params5_b = paras[0, 555072: 555328]
                    layer_params6_w = paras[0, 555328: 1145152]
                    layer_params6_b = paras[0, 1145152: 1145408]
                    layer_params7_w = paras[0, 1145408: 1735232]
                    layer_params7_b = paras[0, 1735232: 1735488]
                    layer_params8_w = paras[0, 1735488:2325312]
                    layer_params8_b = paras[0, 2325312:2325568]
                    layer_params9_w = paras[0, 2325568:2915392]
                    layer_params9_b = paras[0, 2915392:2915648]
                    layer_params10_w = paras[0, 2915648:3505472]
                    layer_params10_b = paras[0, 3505472:3505728]
                    layer_params11_w = paras[0, 3505728:4095552]
                    layer_params11_b = paras[0, 4095552:4095808]
                    layer_params12_w = paras[0, 4095808:4685632]
                    layer_params12_b = paras[0, 4685632:4685888]
                    layer_params13_w = paras[0, 4685888:4711488]
                    layer_params13_b = paras[0, 4711488:4711588]
                    layer_params14_w = paras[0, 4711588:4721588]
                    layer_params14_b = paras[0, 4721588:4721688]
                    last_layer_params_w = paras[0, 4721688:4722688]
                    last_layer_params_b = paras[0, 4722688:4722698]

                    # 根据这些形状，参数调整
                    predicted_parameters['w1'] = np.reshape(layer_params1_w, [3, 3, 3, 64])
                    predicted_parameters['b1'] = np.reshape(layer_params1_b, [64])

                    predicted_parameters['w2'] = np.reshape(layer_params2_w, [3, 3, 64, 64])
                    predicted_parameters['b2'] = np.reshape(layer_params2_b, [64])

                    predicted_parameters['w3'] = np.reshape(layer_params3_w, [3, 3, 64, 128])
                    predicted_parameters['b3'] = np.reshape(layer_params3_b, [128])

                    predicted_parameters['w4'] = np.reshape(layer_params4_w, [3, 3, 128, 128])
                    predicted_parameters['b4'] = np.reshape(layer_params4_b, [128])

                    predicted_parameters['w5'] = np.reshape(layer_params5_w, [3, 3, 128, 256])
                    predicted_parameters['b5'] = np.reshape(layer_params5_b, [256])

                    predicted_parameters['w6'] = np.reshape(layer_params6_w, [3, 3, 256, 256])
                    predicted_parameters['b6'] = np.reshape(layer_params6_b, [256])

                    predicted_parameters['w7'] = np.reshape(layer_params7_w, [3, 3, 256, 256])
                    predicted_parameters['b7'] = np.reshape(layer_params7_b, [256])

                    predicted_parameters['w8'] = np.reshape(layer_params8_w, [3, 3, 256, 256])
                    predicted_parameters['b8'] = np.reshape(layer_params8_b, [256])

                    predicted_parameters['w9'] = np.reshape(layer_params9_w, [3, 3, 256, 256])
                    predicted_parameters['b9'] = np.reshape(layer_params9_b, [256])

                    predicted_parameters['w10'] = np.reshape(layer_params10_w, [3, 3, 256, 256])
                    predicted_parameters['b10'] = np.reshape(layer_params10_b, [256])

                    predicted_parameters['w11'] = np.reshape(layer_params11_w, [3, 3, 256, 256])
                    predicted_parameters['b11'] = np.reshape(layer_params11_b, [256])

                    predicted_parameters['w12'] = np.reshape(layer_params12_w, [3, 3, 256, 256])
                    predicted_parameters['b12'] = np.reshape(layer_params12_b, [256])

                    predicted_parameters['w13'] = np.reshape(layer_params13_w, [256, 100])
                    predicted_parameters['b13'] = np.reshape(layer_params13_b, [100])

                    predicted_parameters['w14'] = np.reshape(layer_params14_w, [100, 100])
                    predicted_parameters['b14'] = np.reshape(layer_params14_b, [100])

                    predicted_parameters['w15'] = np.reshape(last_layer_params_w, [100, 10])
                    predicted_parameters['b15'] = np.reshape(last_layer_params_b, [10])

                    # Send aggregated parameters back to workers
                    for i in range(self.nums_worker):
                        self.queue['queue' + str(i + 2)].put(predicted_parameters)

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
    bsp = BSP(queue)
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
