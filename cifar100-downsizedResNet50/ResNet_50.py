from collections import namedtuple
import tensorflow as tf
import utils
# from seblock import SE_block


HParams = namedtuple('HParams',
                    'batch_size, num_classes, num_residual_units, k, '
                    'initial_lr')


class ResNet18(object):
    def __init__(self, hp, images, labels_a, labels_b, lam):
        self._hp = hp # Hyperparameters
        self._images = images # Input image
        self._labels_a = labels_a
        self._labels_b = labels_b
        self._lam = lam
        # self._global_step = global_step
        # self.is_train = is_train
        # self.regularizer = regularizer

    def build_model(self):
        print('Building model')
        # Init. conv.
        print('\tBuilding unit: init_conv')
        x = utils._conv(self._images, 3, 32, 1, name='init_conv')
        # x = utils._bn(x, self.is_train, name='init_bn')
        x = utils._relu(x)
        x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        # Residual Blocks
        # filters = [32 * self._hp.k, 64 * self._hp.k, 128 * self._hp.k, 16 * self._hp.k, ]
        # strides = [1, 2, 2, 2]

        for i in range(1, 5):

                # First residual unit
                if i == 1:
                    with tf.variable_scope('unit_%d_0' % i) as scope:
                        print('\tBuilding residual unit: %s' % scope.name)
                    shortcut = utils._conv(x, 1, 128, 1, name='shortcut_c1')
                    # Residual
                    x = utils._conv(x, 1, 32, 1, name='conv_11')
                    x = utils._relu(x)
                    x = utils._conv(x, 3, 32, 1, name='conv_12')
                    x = utils._relu(x)
                    x = utils._conv(x, 1, 128, 1, name='conv_13')
                    # Merge
                    x = x + shortcut
                    x = utils._relu(x)
                    # Other residual units

                    for j in range(1, 3):
                        with tf.variable_scope('unit_%d_%d' % (i, j)) as scope:
                            print('\tBuilding residual unit: %s' % scope.name)
                            # Shortcut
                            shortcut = x
                            # Residual
                            x = utils._conv(x, 1, 32, 1, name='conv_1')
                            x = utils._relu(x)
                            x = utils._conv(x, 3, 32, 1, name='conv_2')
                            x = utils._relu(x)
                            x = utils._conv(x, 1, 128, 1, name='conv_3')
                            # Merge
                            x = x + shortcut
                            x = utils._relu(x)

                # Second residual unit
                if i == 2:
                    with tf.variable_scope('unit_%d_0' % i) as scope:
                        print('\tBuilding residual unit: %s' % scope.name)
                    shortcut = utils._conv(x, 1, 256, 2, name='shortcut_c2')
                    x = utils._conv(x, 1, 64, 1, name='conv_21')
                    x = utils._relu(x)
                    x = utils._conv(x, 3, 64, 2, name='conv_22')
                    x = utils._relu(x)
                    x = utils._conv(x, 1, 256, 1, name='conv_23')
                    # Merge
                    x = x + shortcut
                    x = utils._relu(x)

                    for j in range(1, 4):
                        with tf.variable_scope('unit_%d_%d' % (i, j)) as scope:
                            print('\tBuilding residual unit: %s' % scope.name)
                            # Shortcut
                            shortcut = x
                            # Residual
                            x = utils._conv(x, 1, 64, 1, name='conv_1')
                            x = utils._relu(x)
                            x = utils._conv(x, 3, 64, 1, name='conv_2')
                            x = utils._relu(x)
                            x = utils._conv(x, 1, 256, 1, name='conv_3')
                            # Merge
                            x = x + shortcut
                            x = utils._relu(x)

                # Third residual unit
                if i == 3:
                    with tf.variable_scope('unit_%d_0' % i) as scope:
                        print('\tBuilding residual unit: %s' % scope.name)
                    shortcut = utils._conv(x, 1, 512, 2, name='shortcut_c3')
                    x = utils._conv(x, 1, 128, 1, name='conv_31')
                    x = utils._relu(x)
                    x = utils._conv(x, 3, 128, 2, name='conv_32')
                    x = utils._relu(x)
                    x = utils._conv(x, 1, 512, 1, name='conv_33')
                    # Merge
                    x = x + shortcut
                    x = utils._relu(x)

                    for j in range(1, 6):
                        with tf.variable_scope('unit_%d_%d' % (i, j)) as scope:
                            print('\tBuilding residual unit: %s' % scope.name)
                            # Shortcut
                            shortcut = x
                            # Residual
                            x = utils._conv(x, 1, 128, 1, name='conv_1')
                            x = utils._relu(x)
                            x = utils._conv(x, 3, 128, 1, name='conv_2')
                            x = utils._relu(x)
                            x = utils._conv(x, 1, 512, 1, name='conv_3')
                            # Merge
                            x = x + shortcut
                            x = utils._relu(x)

                # Fourth residual unit
                if i == 4:
                    with tf.variable_scope('unit_%d_0' % i) as scope:
                        print('\tBuilding residual unit: %s' % scope.name)
                    shortcut = utils._conv(x, 1, 1024, 2, name='shortcut_c4')
                    x = utils._conv(x, 1, 256, 1, name='conv_41')
                    x = utils._relu(x)
                    x = utils._conv(x, 3, 256, 2, name='conv_42')
                    x = utils._relu(x)
                    x = utils._conv(x, 1, 1024, 1, name='conv_43')
                    # Merge
                    x = x + shortcut
                    x = utils._relu(x)

                    for j in range(1, 3):
                        with tf.variable_scope('unit_%d_%d' % (i, j)) as scope:
                            print('\tBuilding residual unit: %s' % scope.name)
                            # Shortcut
                            shortcut = x
                            # Residual
                            x = utils._conv(x, 1, 256, 1, name='conv_1')
                            x = utils._relu(x)
                            x = utils._conv(x, 3, 256, 1, name='conv_2')
                            x = utils._relu(x)
                            x = utils._conv(x, 1, 1024, 1, name='conv_3')
                            # Merge
                            x = x + shortcut
                            x = utils._relu(x)

        # Last unit
        print(x.shape)
        with tf.variable_scope('avg_pool_layer') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            print(x.shape)

        # Logit
        with tf.variable_scope('full_connected_layer') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.contrib.layers.flatten(x)
            x = utils._fc(x, self._hp.num_classes)

        self.logits = x
        # self._regurization = regurization

        # Probs & preds & acc

    def build_train_op(self):

        cross_entropy_a = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                         labels=tf.argmax(self._labels_a, 1))
        cross_entropy_b = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                         labels=tf.argmax(self._labels_b, 1))
        cross_entropy_mean_a = tf.reduce_mean(cross_entropy_a)
        cross_entropy_mean_b = tf.reduce_mean(cross_entropy_b)
        self.loss = cross_entropy_mean_a * self._lam + cross_entropy_mean_b * (1 - self._lam)
        self.step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(learning_rate=0.0005, global_step=self.step, decay_steps=50,
                                             decay_rate=0.999, staircase=True)
        self.trainable_vars = tf.trainable_variables()
        self.update_vars_opt = self.trainable_vars[0: 46]
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.grads_and_vars = self.opt.compute_gradients(self.loss, var_list=self.update_vars_opt)
        self.update = self.opt.apply_gradients(self.grads_and_vars, global_step=self.step)

        def compute_acc(self):

            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self._labels_a, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def startup_bn(self):

        self.bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)









