from geopy.distance import geodesic
from keras import backend as K

def distance_metric(y_train, y_true):
    return