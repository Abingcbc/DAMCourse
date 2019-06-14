from keras import backend as K
import tensorflow as tf
import numpy as np


def distance_loss(y_pred, y_true):
    return K.mean(K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1)))

def large_mae(y_pred, y_true):
    return K.mean(K.square(y_pred - y_true), axis=-1)*1000
