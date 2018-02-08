import numpy
from sklearn.metrics import confusion_matrix

def _Euclidean_distance(_first_vec,
                        _second_vec):
    return numpy.sqrt(numpy.sum(numpy.square(_first_vec - _second_vec)))

def _calculate_pj(_sample,
                  _means):
    _dists = []
    for _mean in _means:
        _dists.append(_Euclidean_distance(_sample, _mean))
    _dists = numpy.asarray(_dists)
    _prob  = numpy.exp(-_dists)
    _prob  = _prob / numpy.sum(_prob)
    _amax_prob = numpy.argmax(_prob)
    return _prob[_amax_prob], _dists[_amax_prob], _amax_prob

def _merge_data(_data):
    data = numpy.concatenate(tuple(_data))
    return data

def _mapping_labels(_data_y,
                    _mapped_labels):
    _max_labels = len(_mapped_labels)
    data_y      = numpy.ones(_data_y.shape) * _max_labels
    for _count, _mapped_label in enumerate(_mapped_labels):
        _idx = (_data_y == _mapped_label).nonzero()[0]
        data_y[_idx] = _count
    return data_y

def _NCM_unknown(_data_x,
                 _means,
                 _thres):
    pred_label  = []
    _num_classes = len(_means)
    for _count in range(0, len(_data_x)):
        # print '\r|-- Run %d samples' % (_count),

        # Get element
        _sample_x = _data_x[_count]

        _pjxs, _dist, _idx = _calculate_pj(_sample_x,
                                           _means)

        if _dist > _thres:
            pred_label.append(_num_classes)
        else:
            pred_label.append(_idx)
    pred_label = numpy.asarray(pred_label)
    return pred_label

def _calculate_means(_data_x,
                     _data_y,
                     _known_classes):
    means = []
    for _known_class in _known_classes:
        _idx = (_data_y == _known_class).nonzero()[0]
        _sub_data_x = _data_x[_idx,]
        means.append(numpy.mean(_sub_data_x, axis = 0))
    means = numpy.asarray(means, dtype = 'float32')
    return means

def _calculate_confusion_mat(_pred_y,
                             _true_y,
                             _labels = [0,1,2,3,4,5,6,7,8,9]):
    confusion_mat = confusion_matrix(_true_y, _pred_y, labels = _labels)
    return confusion_mat

def _calculate(_confusion_mat):
    diag     = numpy.diag(_confusion_mat)
    accuracy = numpy.sum(diag) * 1.0 / numpy.sum(_confusion_mat)
    sum_col = numpy.sum(_confusion_mat, axis = 0)
    sum_row = numpy.sum(_confusion_mat, axis = 1)
    recall  = diag * 1.0 / (sum_row + 0.0001)
    precis  = diag * 1.0 / (sum_col + 0.0001)
    return accuracy, recall, precis