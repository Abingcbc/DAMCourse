import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras import optimizers
import keras
import matplotlib.pyplot as plt
import utm
from utils import distance_loss, large_mae


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


data = pd.read_csv('hwphase2data_update/new_train.csv')

X = [
    np.array([[mr[4:9],
              mr[9:14],
              mr[14:19],
              mr[19:24],
              mr[24:29],
              mr[29:34]]]).T for mr in data.values
]
X = np.array(X)
zone_number = 0
zone_letter = 0
label = pd.read_csv('hwphase2data_update/label.csv').values
y = []
for i, l in enumerate(label):
    temp = [0 for j in range(357)]
    temp[l[1]] = 1
    y.append(temp)
y = np.array(y)

grid = pd.read_csv('hwphase2data_update/grid.csv').values

history = LossHistory()


def build_model():
    m = Sequential()
    m.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(5, 6, 1)))
    m.add(MaxPooling2D())
    m.add(BatchNormalization())
    m.add(Flatten())
    m.add(Dense(1024, activation='relu'))
    m.add(Dense(512, activation='relu'))
    m.add(Dense(357, activation='softmax'))
    r = optimizers.Adam(lr=0.001)
    m.compile(optimizer=r, loss='categorical_crossentropy', metrics=['accuracy'])
    return m


model = build_model()
model.fit(X, y, epochs=2000, batch_size=1024, callbacks=[history])
history.loss_plot('epoch')
result = model.predict(X[:1])
print(np.argmax(result))
print(model.evaluate(X,y))
