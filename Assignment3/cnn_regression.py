import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from keras import optimizers
from sklearn.model_selection import KFold
import keras
import matplotlib.pyplot as plt
import utm
from utils import distance_loss

'''
936.5715702179168
11.250090223493995
'''


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


data = pd.read_csv('hwphase2data_update/train_2g.csv')

X = [
    np.array([[mr[4:9],
               mr[9:14] if -999 not in mr[9:14] else mr[4:9],
               mr[14:19] if -999 not in mr[14:19] else mr[4:9],
               mr[19:24] if -999 not in mr[19:24] else mr[4:9],
               mr[24:29] if -999 not in mr[24:29] else mr[4:9],
               mr[29:34] if -999 not in mr[29:34] else mr[4:9]]]).T for mr in data.values
]
X = np.array(X)
y = []
zone_number = 0
zone_letter = 0
for mr in data.values:
    lat, lon, zone_number, zone_letter = utm.from_latlon(mr[-4], mr[-5])
    y.append([lat, lon])
y = np.array(y)


def build_model():
    m = Sequential()
    m.add(Conv2D(32, kernel_size=2, activation='relu', input_shape=(5, 6, 1)))
    # m.add(Dropout(0.1))
    # m.add(MaxPooling2D())
    m.add(Conv2D(16, kernel_size=2, activation='relu'))
    m.add(Flatten())
    m.add(Dense(8, activation='relu'))
    m.add(Dense(2))
    r = optimizers.Adam()
    m.compile(optimizer=r, loss=distance_loss, metrics=['mse'])

    return m


history = LossHistory()

# floder = KFold(n_splits=5, random_state=33)
#
# result_mse = []
# result_mae = []
# for train_index, val_index in floder.split(X, y):
#     model = build_model()
#     model.fit(X[train_index], y[train_index], epochs=500, batch_size=128, callbacks=[history])
#     val_mse, val_mae = model.evaluate(X[val_index], y[val_index])
#
#     result_mse.append(val_mse)
#     result_mae.append(val_mae)
#
#
# print(np.mean(result_mse))
# print(np.mean(result_mae))

# model = build_model()
# model.fit(X, y, epochs=800, batch_size=128, callbacks=[history])
# history.loss_plot('epoch')
# result = model.predict(X[:5])
# for i in range(0, 5):
#     print(utm.to_latlon(result[i][0], result[i][1], zone_number, zone_letter))
