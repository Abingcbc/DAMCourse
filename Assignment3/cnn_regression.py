import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import KFold
import keras
import matplotlib.pyplot as plt

'''
936.5715702179168
11.250090223493995
'''

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

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
data = pd.read_csv('data/train_2g.csv')

X = [
    np.array([[mr[3:8],
              mr[8:13] if -999 not in mr[8:13] else mr[3:8],
              mr[13:18] if -999 not in mr[13:18] else mr[3:8],
              mr[18:23] if -999 not in mr[18:23] else mr[3:8],
              mr[23:28] if -999 not in mr[23:28] else mr[3:8],
              mr[28:33] if -999 not in mr[28:33] else mr[3:8]]]).T for mr in data.values
]
X = np.array(X)
y = [mr[-5:-3] for mr in data.values]
y = np.array(y)


def build_model():
    m = Sequential()
    m.add(Conv2D(30, kernel_size=3, activation='relu', input_shape=(5, 6, 1)))
    m.add(Conv2D(10, kernel_size=3, activation='relu'))
    m.add(Flatten())
    m.add(Dense(2))
    m.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return m


floder = KFold(n_splits=5, random_state=33)
history = LossHistory()

result_mse = []
result_mae = []
for train_index, val_index in floder.split(X, y):
    model = build_model()
    model.fit(X[train_index], y[train_index], epochs=100, batch_size=32, callbacks=[history])
    val_mse, val_mae = model.evaluate(X[val_index], y[val_index])

    result_mse.append(val_mse)
    result_mae.append(val_mae)

print(np.mean(result_mse))
print(np.mean(result_mae))
history.loss_plot('epoch')
