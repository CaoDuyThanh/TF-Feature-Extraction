# Load system libs
import timeit
from random import shuffle

# Import Models
from Models.ILTriplet.TripletModel import *

# Import Utils
from Utils.MNistNewDataHelper import *      # Load dataset
from matplotlib import pyplot as plt        # Plot result
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.manifold import TSNE           # 2D Visualization
from Utils.TSBoardHandler import *          # Tensorboard handler
plt.switch_backend('agg')

########################################################################################################################
#                                                                                                                      #
#    CONFIGURATIONS SESSION                                                                                            #
#                                                                                                                      #
########################################################################################################################
# TRAINING HYPER PARAMETER
TRAIN_STATE         = True     # Training state
VALID_STATE         = False    # Validation state
BATCH_SIZE          = 16
NUM_EPOCH           = 1000
LEARNING_RATE       = 0.00001      # Starting learning rate
DISPLAY_FREQUENCY   = 500;         INFO_DISPLAY = '\r%sLearning rate = %f - Epoch = %d - Iter = %d - Cost = %f'
SAVE_FREQUENCY      = 2500
VALIDATE_FREQUENCY  = 5000
VISUALIZE_FREQUENCY = 2500

SAMPLES_PER_CLASS = 50
ALPHA             = 0.2
THRESHOLD         = 1.2

START_EPOCH     = 0
START_ITERATION = 0

# EARLY STOPPING
PATIENCE              = 100000
PATIENCE_INCREASE     = 2
IMRROVEMENT_THRESHOLD = 0.995

# DATASET CONFIGURATION
#DATASET_PATH    = '/net/per920a/export/das14a/satoh-lab/cdthanh/Projects/Dataset/MNIST/Raw/%s/mnist.ckpt.gz'
DATASET_PATH    = '/media/badapple/Data/Projects/Machine Learning/Dataset/MNIST/Raw/%s/mnist.pkl.gz'
# DATASET_PATH    = 'C:/Users/CaoDuyThanh/Downloads/Project/Dataset/MNIST/Raw/%s/mnist.ckpt.gz'

# SPLITTING SETTINGS
SPLITTING_SETS  = [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0], '0,1,2,3,4,5,6,7,8,9-', '0123456789-'],
                   [[0, 1, 2, 3, 5, 6, 9], [4, 7, 8], '0,1,2,3,5,6,9-4,7,8', '0123569-478'],
                   [[0, 2, 3, 5, 6, 9], [1, 4, 7, 8], '0,2,3,5,6,9-1,4,7,8', '023569-1478'],
                   [[0, 2, 3, 6, 9], [1, 4, 5, 7, 8], '0,2,3,6,9-1,4,5,7,8', '02369-14578']]

# STATE PATH
SETTING_PATH      = '../../Pretrained/ILTriplet/Raw/%s/'
TSBOARD_PATH      = SETTING_PATH + 'Tensorboard/'
RECORD_PATH       = SETTING_PATH + 'TripletModel_Record.ckpt'
STATE_PATH        = SETTING_PATH + 'TripletModel_CurrentState.ckpt'
BEST_PREC_PATH    = SETTING_PATH + 'TripletModel_Prec_Best.ckpt'
VISU_KNOWN_PATH   = SETTING_PATH + 'TripletModel_known_class.jpg'
VISU_UNKNOWN_PATH = SETTING_PATH + 'TripletModel_unknown_class.jpg'
VISU_ALL_PATH     = SETTING_PATH + 'TripletModel_all_class.jpg'

#  GLOBAL VARIABLES
dataset       = None
Triplet_model = None
TB_handler    = None

########################################################################################################################
#                                                                                                                      #
#    LOAD DATASET SESSIONS                                                                                             #
#                                                                                                                      #
########################################################################################################################
def _load_dataset(_all_path):
    global dataset

    dataset = MNistNewDataHelper(_dataset_path = _all_path['dataset_path'])
    print ('|-- Load path = %s ! Completed !' % (_all_path['dataset_path']))

########################################################################################################################
#                                                                                                                      #
#    CREATE TRIPLET MODEL                                                                                              #
#                                                                                                                      #
########################################################################################################################
def _create_Triplet_model():
    global Triplet_model
    Triplet_model = TripletModel()

########################################################################################################################
#                                                                                                                      #
#    CREATE TENSOR BOARD                                                                                               #
#                                                                                                                      #
########################################################################################################################
def _create_TSBoard_model(_all_path):
    global TB_hanlder
    TB_hanlder = TSBoardHandler(_save_path = _all_path['tsboard_path'])

########################################################################################################################
#                                                                                                                      #
#    UTILITIES ..........                                                                                              #
#                                                                                                                      #
########################################################################################################################
def _split_data(_data,
                _ratios):
    splitted_data = []
    _num_samples  = len(_data)
    for _idx in range(len(_ratios) - 1):
        splitted_data.append(_data[int(_ratios[_idx]     * _num_samples) :
                                   int(_ratios[_idx + 1] * _num_samples), ])
    
    return splitted_data

def _sort(_data,
          _label):
    _idx = numpy.argsort(_label)
    sorted_data  = _data[_idx, ]
    sorted_label = _label[_idx, ]
    return sorted_data, sorted_label

def _sample_data(_set_x,
                 _set_y,
                 _labels):
    sampled_set_x      = []
    sampled_set_y      = []
    nsampled_per_class = []
    for _label in _labels:
        _sub_set_idx = (_set_y == _label).nonzero()[0]
        shuffle(_sub_set_idx)
        _sub_set_idx = _sub_set_idx[ : SAMPLES_PER_CLASS,]
        
        sampled_set_x.append(_set_x[_sub_set_idx])
        sampled_set_y.append(_set_y[_sub_set_idx])
        nsampled_per_class.append(len(_sub_set_idx))
    sampled_set_x      = numpy.concatenate(tuple(sampled_set_x), axis = 0)
    sampled_set_y      = numpy.concatenate(tuple(sampled_set_y), axis = 0)
    return sampled_set_x, sampled_set_y, nsampled_per_class

def _extract_feature(_session,
                     _model,
                     _set_x):
    _num_sample = len(_set_x)
    _num_batch  = int(numpy.ceil(_num_sample * 1.0 / BATCH_SIZE))
    feature = []
    for _batch_id in range(_num_batch):
        _sub_set_x = _set_x[_batch_id      * BATCH_SIZE :
                           (_batch_id + 1) * BATCH_SIZE, ]
        _result = _model.feat_ext_func(_session = _session,
                                       _state   = VALID_STATE,
                                       _batch_x = _sub_set_x)
        feature.append(_result[0])
    feature = numpy.concatenate(tuple(feature), axis = 0)
    return feature

def _select_triplet(_set_data,
                    _set_feature,
                    _set_label,
                    _nsampled_per_class,
                    _labels):
    trip_idx        = 0
    emb_start_idx   = 0
    num_trips       = 0
    triplets_anchor = []
    triplets_pos    = []
    triplets_neg    = []

    for i in xrange(len(_labels)):
        nrof_images = int(_nsampled_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = numpy.sum(numpy.square(_set_feature[a_idx,] - _set_feature), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = numpy.sum(numpy.square(_set_feature[a_idx,] - _set_feature[p_idx,]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = numpy.NaN
                all_neg = numpy.where(neg_dists_sqr - pos_dist_sqr < ALPHA)[0]
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = numpy.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets_anchor.append(_set_data[a_idx,])
                    triplets_pos.append(_set_data[p_idx,])
                    triplets_neg.append(_set_data[n_idx,])
                    trip_idx += 1
                num_trips += 1
        emb_start_idx += nrof_images

    if len(triplets_anchor) is 0:
        return []

    triplets_anchor = numpy.asarray(triplets_anchor, dtype = 'float32')
    triplets_pos    = numpy.asarray(triplets_pos, dtype = 'float32')
    triplets_neg    = numpy.asarray(triplets_neg, dtype = 'float32')
    triplets        = numpy.concatenate((triplets_anchor, triplets_pos, triplets_neg), axis = 1)
    numpy.random.shuffle(triplets)
    triplets        = triplets.reshape((triplets_anchor.shape[0] * 3, 1, triplets_anchor.shape[2], triplets_anchor.shape[3]))

    return triplets

########################################################################################################################
#                                                                                                                      #
#    VALID FEATURE EXTRACTION MODEL..........                                                                          #
#                                                                                                                      #
########################################################################################################################
import scipy
import NCMClassifier
def _valid_model(_session,
                 _model,
                 _valid_data,
                 _known_classes):
    # --- Extract info ---
    _valid_set_x, _valid_set_y = _valid_data
    _valid_new_data_y = NCMClassifier._mapping_labels(_valid_set_y, _known_classes)
    _valid_feature_x  = _extract_feature(_session,
                                         _model,
                                         _valid_set_x)
    _means            = NCMClassifier._calculate_means(_data_x        = _valid_feature_x,
                                                       _data_y        = _valid_set_y,
                                                       _known_classes = _known_classes)
    best_acc_valid = 0
    best_thres     = 0
    for _thres in numpy.arange(0.1, 2.0, 0.01):
        print '\r|-- Thres = %f - Best thres = %f - Best acc = %f' % (_thres, best_thres, best_acc_valid),

        _pred_label = NCMClassifier._NCM_unknown(_data_x = _valid_feature_x,
                                                 _means  = _means,
                                                 _thres  = _thres)
        _confusion_mat = NCMClassifier._calculate_confusion_mat(_pred_label, _valid_new_data_y, [0,1,2,3,4,5,6,7,8,9,10])
        _accuracy,_,_  = NCMClassifier._calculate(_confusion_mat)

        if _accuracy > best_acc_valid:
            best_acc_valid = _accuracy
            best_thres     = _thres

    return best_acc_valid, best_thres

def _merge_data(_data):
    data = numpy.concatenate(tuple(_data))
    return data

########################################################################################################################
#                                                                                                                      #
#    TRAIN FEATURE EXTRACTION MODEL..........                                                                          #
#                                                                                                                      #
########################################################################################################################
def _train_test_model(_all_path,
                      _known_classes,
                      _unknown_classes):
    global dataset, \
           Triplet_model, \
           TB_handler
    # ===== Prepare path =====
    _setting_path    = _all_path['setting_path']
    _record_path     = _all_path['record_path']
    _state_path      = _all_path['state_path']
    _visu_known_path = _all_path['visu_known_classes']
    _best_prec_path  = _all_path['best_prec_path']
           
    # ===== Prepare dataset =====
    # ----- Train data -----
    _train_known_data_x = dataset.train_known_data_x
    _train_known_data_y = dataset.train_known_data_y
    _train_known_data_x, \
    _train_known_data_y = _sort(_train_known_data_x, _train_known_data_y)

    # ----- Valid data -----
    _valid_known_data_x = dataset.valid_known_data_x
    _valid_known_data_y = dataset.valid_known_data_y
    _valid_known_data_x, \
    _valid_known_data_y = _sort(_valid_known_data_x, _valid_known_data_y)

    _valid_unknown_data_x = dataset.valid_unknown_data_x
    _valid_unknown_data_y = dataset.valid_unknown_data_y
    _valid_data_x = NCMClassifier._merge_data([_valid_known_data_x, _valid_unknown_data_x])
    _valid_data_y = NCMClassifier._merge_data([_valid_known_data_y, _valid_unknown_data_y])

    # ===== Start session =====
    _config  = tf.ConfigProto()
    _config.gpu_options.allow_growth = True
    _session = tf.Session(config = _config)

    # ----- Initialize params -----
    _session.run(tf.global_variables_initializer())
    _session.run(tf.local_variables_initializer())

    # ----- Save graph -----
    TB_hanlder.save_graph(_graph = _session.graph)

    # ===== Load data record =====
    print ('|-- Load previous record !')
    iter_train_record = []
    cost_train_record = []
    iter_valid_record = []
    prec_valid_record = []
    best_prec_valid   = 0
    best_thres        = 0
    _epoch         = START_EPOCH
    _iter          = START_ITERATION
    _learning_rate = LEARNING_RATE
    if check_file_exist(_record_path, _throw_error = False):
        _file = open(_record_path, 'rb')
        iter_train_record = pickle.load(_file)
        cost_train_record = pickle.load(_file)
        iter_valid_record = pickle.load(_file)
        prec_valid_record = pickle.load(_file)
        best_prec_valid   = pickle.load(_file)
        best_thres        = pickle.load(_file)
        _epoch            = pickle.load(_file)
        _iter             = pickle.load(_file)
        _learning_rate    = pickle.load(_file)
        _file.close()
    print ('|-- Load previous record ! Completed !')

    # ===== Load state =====
    _saver = tf.train.Saver()
    print ('|-- Load state !')
    if tf.train.checkpoint_exists(_state_path):
        _saver.restore(sess      = _session,
                       save_path = _state_path)
    print ('|-- Load state ! Completed !')

    # ===== Training start =====
    # ----- Temporary record -----
    _cost_train_temp = []
    _ratios          = []

    # ----- Train -----
    while (_epoch < NUM_EPOCH):
        _epoch += 1
        
        # --- Sample training data ---
        _sampled_train_data, \
        _sampled_train_label, \
        _nsampled_per_class = _sample_data(_train_known_data_x, _train_known_data_y, _known_classes)
        
        # --- Extract feature ---
        _sampled_train_feature = _extract_feature(_session,
                                                  Triplet_model,
                                                  _sampled_train_data)
        
        # --- Select triplets ---
        _triplet_set_x = _select_triplet(_sampled_train_data,
                                         _sampled_train_feature, 
                                         _sampled_train_label, 
                                         _nsampled_per_class,
                                         _known_classes)

        if len(_triplet_set_x) < 1:
            _done_looping = True

        # --- Train triplets ---
        _num_batch_trained_data = len(_triplet_set_x) / (3 * BATCH_SIZE)
        for _id_batch_trained_data in range(_num_batch_trained_data):
            _train_start_time = timeit.default_timer()
            _train_batch_x = _triplet_set_x[_id_batch_trained_data      * BATCH_SIZE * 3 :
                                           (_id_batch_trained_data + 1) * BATCH_SIZE * 3, ]
            
            _iter += 1
            _result = Triplet_model.feat_train_func(_session       = _session,
                                                    _state         = TRAIN_STATE,
                                                    _learning_rate = _learning_rate,
                                                    _batch_size    = BATCH_SIZE,
                                                    _batch_x       = _train_batch_x,
                                                    _alpha         = ALPHA)
            # Temporary save info
            _cost_train_temp.append(_result[0])
            _ratios.append(_result[1])
            _train_end_time = timeit.default_timer()
            
            # Print information
            print '\r|-- Trained %d / %d batch - Time = %f' % (_id_batch_trained_data, _num_batch_trained_data, _train_end_time - _train_start_time),

            if _iter % DISPLAY_FREQUENCY == 0:
                # Print information of current training in progress
                print (INFO_DISPLAY % ('|-- ', _learning_rate, _epoch, _iter, numpy.mean(_cost_train_temp)))
                iter_train_record.append(_iter)
                cost_train_record.append(numpy.mean(_cost_train_temp))
                print ('|-- Ratio = %f' % (numpy.mean(_ratios)))

                # Add summary
                TB_hanlder.log_scalar(_name_scope = 'Metadata',
                                      _name       = 'Learning rate',
                                      _value      = _learning_rate,
                                      _step       = _iter)
                TB_hanlder.log_scalar(_name_scope = 'Train',
                                      _name       = 'Loss',
                                      _value      = numpy.mean(_cost_train_temp),
                                      _step       = _iter)
                TB_hanlder.log_scalar(_name_scope = 'Train',
                                      _name       = 'Ratio',
                                      _value      = numpy.mean(_ratios),
                                      _step       = _iter)

                # Reset list
                _cost_train_temp  = []

            if _iter % SAVE_FREQUENCY == 0:
                # Save record
                _file = open(_record_path, 'wb')
                pickle.dump(iter_train_record, _file, 2)
                pickle.dump(cost_train_record, _file, 2)
                pickle.dump(iter_valid_record, _file, 2)
                pickle.dump(prec_valid_record, _file, 2)
                pickle.dump(best_prec_valid, _file, 2)
                pickle.dump(best_thres, _file, 2)
                pickle.dump(_epoch, _file, 2)
                pickle.dump(_iter, _file, 2)
                pickle.dump(_learning_rate, _file, 2)
                _file.close()
                print ('+ Save record ! Completed !')
    
                # Save state
                _saver.save(sess      = _session,
                            save_path = _state_path)
                print ('+ Save state ! Completed !')

            if _iter % VALIDATE_FREQUENCY == 0:
                print ('\n------------------- Validate Model -------------------')
                _prec_valid, _thres_valid = _valid_model(_session       = _session,
                                                         _model         = Triplet_model,
                                                         _valid_data    = [_valid_data_x, _valid_data_y],
                                                         _known_classes = _known_classes)
                iter_valid_record.append(_iter)
                prec_valid_record.append(_prec_valid)
                print ('\n+ Validate model finished! Prec = %f - Thres = %f' % (_prec_valid, _thres_valid))
                print ('+ Validate model finished! Best Prec = %f - Best Thres = %f' % (best_prec_valid, best_thres))
                print ('\n------------------- Validate Model (Done) -------------------')

                # Add summary
                TB_hanlder.log_scalar(_name_scope = 'Valid',
                                      _name       = 'Accuracy',
                                      _value      = _prec_valid,
                                      _step       = _iter)

                # Save model if its cost better than old one
                if (_prec_valid > best_prec_valid):
                    best_prec_valid = _prec_valid
                    best_thres      = _thres_valid

                    # Save best model
                    _saver.save(sess      = _session,
                                save_path = _best_prec_path)
                    print ('+ Save best prec model ! Complete !')

            if _iter % VISUALIZE_FREQUENCY == 0:
                print ('\n------------------- Visualize Model -------------------')
                _train_known_image = _visualize_model(_session     = _session,
                                                      _model       = Triplet_model,
                                                      _test_set_x  = _valid_known_data_x,
                                                      _test_set_y  = _valid_known_data_y,
                                                      _num_samples = 100)
                _train_unknown_image = _visualize_model(_session     = _session,
                                                        _model       = Triplet_model,
                                                        _test_set_x  = _valid_unknown_data_x,
                                                        _test_set_y  = _valid_unknown_data_y,
                                                        _num_samples = 100)
                _train_all_image = _visualize_model(_session     = _session,
                                                    _model       = Triplet_model,
                                                    _test_set_x  = _valid_data_x,
                                                    _test_set_y  = _valid_data_y,
                                                    _num_samples = 100)
                # Add summary
                TB_hanlder.log_images(_name_scope = 'Valid',
                                      _name       = 'Visualization',
                                      _images     = [_train_known_image, _train_unknown_image, _train_all_image],
                                      _step       = _iter,
                                      _is_gray    = False)
                print ('\n------------------- Visualize Model -------------------')
    _session.close()
            
def _visualize_model(_session,
                     _model,
                     _test_set_x,
                     _test_set_y,
                     _num_samples):
    # ===== Create feature =====
    # --- Get sub set ---
    idx = range(len(_test_set_x))
    shuffle(idx)
    idx = idx[:_num_samples]
    _test_set_x = _test_set_x[idx,]
    _test_set_y = _test_set_y[idx,]

    # --- Extract features from sub set ---
    features = _extract_feature(_session, _model, _test_set_x)
    labels   = _test_set_y

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    # plot the result
    vis_x = tsne_results[:, 0]
    vis_y = tsne_results[:, 1]

    fig, ax = plt.subplots()
    cax = ax.scatter(vis_x, vis_y, c = labels, cmap = plt.cm.get_cmap('tab10', 10), marker = '+')
    fig.colorbar(cax, ticks=range(10))
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = numpy.fromstring(canvas.tostring_rgb(), dtype = 'uint8').reshape(int(height), int(width), 3)
    plt.gcf().clear()

    return img

def _evaluate_model(_all_path,
                    _known_classes,
                    _unknown_classes,
                    _num_samples = 7000):
    global dataset, \
           Triplet_model
    print ('------------------- Evaluate Model -------------------')
    # ===== Prepare path =====
    _setting_path      = _all_path['setting_path']
    _state_path        = _all_path['state_path']
    _visu_known_path   = _all_path['visu_known_classes']
    _visu_unknown_path = _all_path['visu_unknown_classes']
    _visu_all_path     = _all_path['visu_all_classes']
    _log_file_path     = _setting_path + 'report.txt'

    # ===== Prepare dataset =====
    _valid_known_data_x   = dataset.valid_known_data_x
    _valid_known_data_y   = dataset.valid_known_data_y
    _valid_unknown_data_x = dataset.valid_unknown_data_x
    _valid_unknown_data_y = dataset.valid_unknown_data_y
    _valid_data_x = _merge_data([_valid_known_data_x, _valid_unknown_data_x])
    _valid_data_y = _merge_data([_valid_known_data_y, _valid_unknown_data_y])
           
    # ===== Load best state =====
    print ('|-- Load best model !')
    if check_file_exist(_state_path, _throw_error = False):
        _file = open(_state_path, 'rb')
        Triplet_model.load_model(_file)
        _file.close()
    print ('|-- Load best model ! Completed !')

    # # ===== Evaluate dataset =====
    # _prec_unknown_set = _valid_model(_model=Triplet_model,
    #                                  _valid_data=[dataset.unknown_set_x,
    #                                               dataset.unknown_set_y])
    # _file = open(_log_file_path, 'w')
    # _file.writelines('Precision of Unknown set = %f \n' % (_prec_unknown_set))
    #
    # _prec_known_set = _valid_model(_model=Triplet_model,
    #                                _valid_data=[dataset.known_set_x,
    #                                             dataset.known_set_y])
    # _file.writelines('Precision of Known set = %f \n' % (_prec_known_set))
    #
    # _prec_all_set = _valid_model(_model=Triplet_model,
    #                              _valid_data=[dataset.all_set_x,
    #                                           dataset.all_set_y])
    # _file.writelines('Precision of All set = %f \n' % (_prec_all_set))
    # _file.close()

    # ===== Visualize dataset =====
    # --- Visualize unknown classes ---
    _visualize_model(_model      = Triplet_model,
                     _test_set_x = _valid_unknown_data_x,
                     _test_set_y = _valid_unknown_data_y,
                     _visu_path  = _visu_unknown_path,
                     _num_samples = _num_samples)

    # --- Visualize known classes ---
    _visualize_model(_model      = Triplet_model,
                     _test_set_x = _valid_known_data_x,
                     _test_set_y = _valid_known_data_y,
                     _visu_path  = _visu_known_path,
                     _num_samples = _num_samples)

    # --- Visualize all classes ---
    _visualize_model(_model      = Triplet_model,
                     _test_set_x = _valid_data_x,
                     _test_set_y = _valid_data_y,
                     _visu_path   = _visu_all_path,
                     _num_samples = _num_samples)
    
    print ('------------------- Evaluate Model (Done) -------------------')

def _draw_heatmap(_all_path,
                  _known_classes,
                  _unknown_classes):
    global dataset, \
           Triplet_model

    # ===== Prepare path =====
    _setting_path = _all_path['setting_path']
    _state_path   = _all_path['state_path']

    # ===== Load best state =====
    print ('|-- Load best model !')
    if check_file_exist(_state_path, _throw_error=False):
        _file = open(_state_path, 'rb')
        Triplet_model.load_model(_file)
        _file.close()
    print ('|-- Load best model ! Completed !')

    # ===== Create feature =====
    _all_labels = _known_classes + _unknown_classes
    _set_x = dataset.all_set_x
    _set_y = dataset.all_set_y

    # ===== Extract feature =====
    for _label in _all_labels:
        _idx       = (_set_y == _label).nonzero()[0]
        _sub_set_x = _set_x[_idx, ][:256,]
        _feature   = _extract_feature(Triplet_model, _sub_set_x)

        plt.imshow(_feature.transpose(), cmap='hot', interpolation='nearest')
        plt.savefig(_setting_path + 'heatmap_class_%d.jpg' % _label)
        plt.gcf().clear()

def _train_each_setting():
    count = 0
    for _setting in SPLITTING_SETS:
        if count < 1:
            count += 1
            continue
        _known_classes, \
        _unknown_classes, \
        _name_setting, \
        _name_save = _setting

        _all_path                   = dict()
        _all_path['dataset_path']   = DATASET_PATH % _name_setting
        _all_path['setting_path']   = SETTING_PATH % _name_save;     check_path_and_create(_all_path['setting_path'])
        _all_path['tsboard_path']   = TSBOARD_PATH % _name_save;     check_path_and_create(_all_path['tsboard_path'])
        _all_path['record_path']    = RECORD_PATH % _name_save
        _all_path['state_path']     = STATE_PATH % _name_save
        _all_path['best_prec_path'] = BEST_PREC_PATH % _name_save
        _all_path['visu_known_classes']   = VISU_KNOWN_PATH % _name_save
        _all_path['visu_unknown_classes'] = VISU_UNKNOWN_PATH % _name_save
        _all_path['visu_all_classes']     = VISU_ALL_PATH % _name_save

        _load_dataset(_all_path = _all_path)
        _create_Triplet_model()
        _create_TSBoard_model(_all_path = _all_path)
        _train_test_model(_all_path        = _all_path,
                          _known_classes   = _known_classes,
                          _unknown_classes = _unknown_classes)
        tf.reset_default_graph()  # Reset graph
        # _evaluate_model(_all_path        = _all_path,
        #                 _known_classes   = _known_classes,
        #                 _unknown_classes = _unknown_classes)
        # _draw_heatmap(_all_path        = _all_path,
        #               _known_classes   = _known_classes,
        #               _unknown_classes = _unknown_classes)
        
if __name__ == '__main__':
    _train_each_setting()