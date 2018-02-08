import tensorflow as tf
from Models.ILTriplet.EncoderModel import *
from Models.ILTriplet.DecoderModel import *
from collections import OrderedDict
from Layers.Optimizer import *

class TripletAEModel():
    def __init__(self):
        # ===== Create tensor variables to store input / output data =====
        self.input_feat    = tf.placeholder(tf.float32, shape = [None, 1, 28, 28], name = 'input_feat')
        self.input_recon   = tf.placeholder(tf.float32, shape = [None, 1, 28, 28], name = 'input_recon')
        self.Alpha         = tf.placeholder(tf.float32, shape = (), name = 'alpha')
        self.batch_size    = tf.placeholder(tf.int32, shape = (), name = 'batch_size')
        self.state         = tf.placeholder(tf.bool, shape = (), name = 'state')
        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'learning_rate')

        # ===== Create model =====
        # ----- Create net -----
        self.net_name = 'CNN for Feature Extraction'
        self.layers   = OrderedDict()

        # ----- Reshape input -----
        self.layers['input_feat']  = tf.reshape(self.input_feat, [-1, 28, 28, 1])
        self.layers['input_recon'] = tf.reshape(self.input_recon, [-1, 28, 28, 1])

        # ----- Feature Extraction Model -----
        self.feat_net = EncoderModel(_state  = self.state,
                                     _input  = self.layers['input_feat'])
        _feat_out     = self.feat_net.get_layer('st3_fc')
        _feat_params  = self.feat_net.params

        # ----- Encoder | Decoder Model -----
        self.encoder_net = self.feat_net.clone(_state  = self.state,
                                               _input  = self.layers['input_recon'])
        _encoder_out     = self.encoder_net.get_layer('st3_fc')

        self.decoder_net = DecoderModel(_state  = self.state,
                                        _input  = _encoder_out)
        _decoder_out     = self.decoder_net.get_layer('st3_sig')
        _decoder_params  = self.decoder_net.params

        # ===== Cost function =====
        # ----- Triplet -----
        # --- Cost ---
        def norm_feature(_feature):
            _dist = tf.sqrt(tf.reduce_sum(_feature ** 2, axis = 1, keep_dims = True)) + 0.00001
            return _feature / _dist
        _feat_norm        = norm_feature(_feat_out)
        _feature_anchor   = _feat_norm[ 0 : : 3, ]
        _feature_positive = _feat_norm[ 1 : : 3, ]
        _feature_negative = _feat_norm[ 2 : : 3, ]
        _feat_cost        = tf.reduce_mean(
                                tf.sqrt(tf.reduce_sum(tf.square(_feature_anchor - _feature_positive), axis = 1)) \
                                - tf.sqrt(tf.reduce_sum(tf.square(_feature_anchor - _feature_negative), axis = 1)) \
                                + self.Alpha
                            )
        _recon_cost       = tf.reduce_sum(tf.square(_decoder_out - self.layers['input_recon'])) / tf.cast(self.batch_size, tf.float32)
        _cost = _feat_cost + 0.392 * _recon_cost

        # --- Optimizer ---
        _adam_opti  = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        _params     = _feat_params + _decoder_params
        _grads      = tf.gradients(_cost, _params)
        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops):
            self.optimizer = Optimizer(_optimizer_opt = _adam_opti,
                                       _grads         = _grads,
                                       _params        = _params)

        # ===== Function =====
        # ----- Feature -----
        def _feat_ext_func(_session, _state, _batch_x):
            return _session.run([_feat_norm],
                                feed_dict = {
                                    'state:0':      _state,
                                    'input_feat:0': _batch_x,
                                })
        self.feat_ext_func = _feat_ext_func

        def _feat_train_func(_session, _state, _learning_rate, _batch_size,
                             _batch_x, _alpha, _batch_recon_x):
            return _session.run([_feat_cost, _recon_cost, self.optimizer.ratio, self.optimizer.train_opt],
                                feed_dict = {
                                    'state:0':         _state,
                                    'learning_rate:0': _learning_rate,
                                    'batch_size:0':    _batch_size,
                                    'input_feat:0':    _batch_x,
                                    'alpha:0':         _alpha,
                                    'input_recon:0':   _batch_recon_x,
                                })
        self.feat_train_func = _feat_train_func

        def _recon_func(_session, _state,
                        _batch_recon_x):
            return _session.run([_decoder_out],
                                feed_dict = {
                                    'state:0':         _state,
                                    'input_recon:0':   _batch_recon_x,
                                })
        self.recon_func = _recon_func
