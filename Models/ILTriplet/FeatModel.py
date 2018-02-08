import tensorflow as tf
from collections import OrderedDict

class FeatModel:
    def __init__(self,
                 _state,
                 _input,
                 _reuse = False):
        # ===== Create tensor variables to store input / output data =====
        self.state = _state
        self.X     = _input

        # ===== Create model =====
        # ----- Create net -----
        self.net_name = 'CNN for Feature Extraction'
        self.layers   = OrderedDict()

        # ----- Stack 1 -----
        with tf.variable_scope('simple_net', reuse = _reuse):
            with tf.variable_scope('stack1'):
                # --- Convolution ---
                self.layers['st1_conv'] = tf.layers.conv2d(inputs      = self.X,
                                                           filters     = 32,
                                                           kernel_size = [5, 5],
                                                           strides     = [2, 2],
                                                           padding     = 'same',
                                                           activation  = None,
                                                           name        = 'conv')

                # --- Batch normalization ---
                self.layers['st1_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st1_conv'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st1_relu'] = tf.nn.relu(features = self.layers['st1_batchnorm'],
                                                     name     = 'relu')

            # ----- Stack 2 -----
            with tf.variable_scope('stack2'):
                # --- Convolution ---
                self.layers['st2_conv'] = tf.layers.conv2d(inputs      = self.layers['st1_relu'],
                                                           filters     = 64,
                                                           kernel_size = [5, 5],
                                                           strides     = [2, 2],
                                                           padding     = 'same',
                                                           activation  = None,
                                                           name        = 'conv')

                # --- Batch normalization ---
                self.layers['st2_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st2_conv'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st2_relu']      = tf.nn.relu(features = self.layers['st2_batchnorm'],
                                                          name     = 'relu')

            # ----- Stack 3 -----
            with tf.variable_scope('stack3'):
                # --- Reshape ---
                self.layers['st3_reshape'] = tf.reshape(self.layers['st2_relu'], [-1, 7 * 7 * 64])

                # --- Fully Connected Layer ---
                self.layers['st3_fc'] = tf.layers.dense(inputs     = self.layers['st3_reshape'],
                                                        units      = 64,
                                                        activation = None)

        # --- Params ---
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'simple_net')

    def get_layer(self,
                  _layer_name):
        return self.layers[_layer_name]

    def clone(self,
              _state,
              _input):
        return FeatModel(_state = _state,
                         _input = _input,
                         _reuse = True)
