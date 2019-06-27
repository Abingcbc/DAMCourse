from keras import backend as K
import tensorflow as tf
import numpy as np


def distance_loss(y_pred, y_true):
    return K.sqrt(K.mean(K.sum(K.square(y_pred-y_true), axis=-1)))

def median_absolute_deviation(y_pred, y_true):
    deviation = np.abs(y_pred-y_true)
    return np.median(deviation, axis=0)
