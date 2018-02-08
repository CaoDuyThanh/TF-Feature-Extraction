import tensorflow as tf
from collections import OrderedDict

class DecoderModel:
    def __init__(self,
                 _state,
                 _input,
                 _reuse = False):
        # ===== Create tensor variables to store input / output data =====
        self.state = _state
        self.X     = _input

        # ===== Create model =====
        # ----- Create net -----
        self.net_name = 'Decoder net'
        self.layers   = OrderedDict()

        # ----- Stack 1 -----
        with tf.variable_scope('decoder_net', reuse = _reuse):
            with tf.variable_scope('stack1'):
                # --- Fully connected ---
                self.layers['st1_fc'] = tf.layers.dense(inputs     = self.X,
                                                        units      = 7 * 7 * 64,
                                                        activation = None)

                # --- Batch normalization ---
                self.layers['st1_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st1_fc'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st1_relu'] = tf.nn.relu(features = self.layers['st1_batchnorm'],
                                                     name     = 'relu')

                # --- Reshape ---
                self.layers['st1_reshape'] = tf.reshape(self.layers['st1_relu'], [-1, 7, 7, 64])


            # ----- Stack 2 -----
            with tf.variable_scope('stack2'):
                # --- Transposed Convolution ---
                self.layers['st2_trans_conv'] = tf.layers.conv2d_transpose(inputs      = self.layers['st1_reshape'],
                                                                           filters     = 32,
                                                                           kernel_size = [5, 5],
                                                                           strides     = [2, 2],
                                                                           padding     = 'same',
                                                                           activation  = None,
                                                                           name        = 'conv')

                # --- Batch normalization ---
                self.layers['st2_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st2_trans_conv'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st2_relu']      = tf.nn.relu(features = self.layers['st2_batchnorm'],
                                                          name     = 'relu')

            # ----- Stack 3 -----
            with tf.variable_scope('stack3'):
                # --- Transposed Convolution ---
                self.layers['st3_trans_conv'] = tf.layers.conv2d_transpose(inputs      = self.layers['st2_relu'],
                                                                           filters     = 1,
                                                                           kernel_size = [5, 5],
                                                                           strides     = [2, 2],
                                                                           padding     = 'same',
                                                                           activation  = None,
                                                                           name        = 'conv')

                # --- Sigmoid ---
                self.layers['st3_sig'] = tf.nn.sigmoid(x    = self.layers['st3_trans_conv'],
                                                       name = 'sigmoid')

        # --- Params ---
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'decoder_net')

    def get_layer(self,
                  _layer_name):
        return self.layers[_layer_name]

    def clone(self,
              _state,
              _input):
        return DecoderModel(_state = _state,
                            _input = _input,
                            _reuse = True)
