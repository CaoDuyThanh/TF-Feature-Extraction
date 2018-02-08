import tensorflow as tf
from Models.ILTriplet.FeatModel import *
from collections import OrderedDict
from Layers.Optimizer import *

class TripletModel():
    def __init__(self):
        # ===== Create tensor variables to store input / output data =====
        self.input         = tf.placeholder(tf.float32, shape = [None, 1, 28, 28], name = 'input')
        self.Alpha         = tf.placeholder(tf.float32, shape = (), name = 'alpha')
        self.batch_size    = tf.placeholder(tf.int32, shape = (), name = 'batch_size')
        self.state         = tf.placeholder(tf.bool, shape = (), name = 'state')
        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'learning_rate')

        # ===== Create model =====
        # ----- Create net -----
        self.net_name = 'CNN for Feature Extraction'
        self.layers   = OrderedDict()

        # ----- Reshape input -----
        self.layers['input'] = tf.reshape(self.input, [-1, 28, 28, 1])

        # ----- Feature Extraction Model -----
        self.featext_net = FeatModel(_state  = self.state,
                                     _input  = self.layers['input'])
        _feature         = self.featext_net.get_layer('st3_fc')
        _feature_params  = self.featext_net.params

        # ===== Cost function =====
        # ----- Triplet -----
        # --- Cost ---
        def norm_feature(_feature):
            _dist = tf.sqrt(tf.reduce_sum(_feature ** 2, axis = 1, keep_dims = True)) + 0.00001
            return _feature / _dist
        _feature_norm     = norm_feature(_feature)
        _feature_anchor   = _feature_norm[ 0 : : 3, ]
        _feature_positive = _feature_norm[ 1 : : 3, ]
        _feature_negative = _feature_norm[ 2 : : 3, ]
        _cost             =   tf.sqrt(tf.reduce_sum(tf.square(_feature_anchor - _feature_positive), axis = 1)) \
                            - tf.sqrt(tf.reduce_sum(tf.square(_feature_anchor - _feature_negative), axis = 1)) \
                            + self.Alpha
        _cost = tf.reduce_mean(_cost)
        
        # --- Optimizer ---
        _adam_opti  = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        _grads      = tf.gradients(_cost, _feature_params)
        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops):
            self.optimizer = Optimizer(_optimizer_opt = _adam_opti,
                                       _grads         = _grads,
                                       _params        = _feature_params)
        
        # ===== Function =====
        # ----- Feature -----
        def _feat_ext_func(_session, _state, _batch_x):
            return _session.run([_feature_norm],
                                feed_dict = {
                                    'state:0': _state,
                                    'input:0': _batch_x,
                                })
        self.feat_ext_func = _feat_ext_func

        def _feat_train_func(_session, _state, _learning_rate, _batch_size,
                             _batch_x, _alpha):
            return _session.run([_cost, self.optimizer.ratio, self.optimizer.train_opt],
                                feed_dict = {
                                    'state:0':         _state,
                                    'learning_rate:0': _learning_rate,
                                    'batch_size:0':    _batch_size,
                                    'input:0':         _batch_x,
                                    'alpha:0':         _alpha,
                                })
        self.feat_train_func = _feat_train_func
