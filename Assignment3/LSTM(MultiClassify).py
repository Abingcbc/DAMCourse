import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers import Masking
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, TimeDistributed
from keras.optimizers import RMSprop
from keras.layers import TimeDistributed

import keras
import matplotlib.pyplot as plt
import utm
from sklearn.model_selection import KFold
from sklearn.preprocessing import minmax_scale
from math import ceil

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


data = pd.read_csv('/Users/cbc/Project/Python/DAMCourse/Assignment3/hwphase2data_update/new_formatted_data.csv')

minmax_scale(data, copy=False)
X = [
    np.concatenate([mr[4:9],mr[9:14],mr[14:19],mr[19:24],mr[24:29],
               mr[29:34]]) for mr in data[data['IMSI'==460012796993062.0]].values
]
X = np.array(X)
zone_number = 0
zone_letter = 0
label = pd.read_csv('hwphase2data_update/label1.csv')[data['IMSI'==460012796993062.0]].values
y = []
for i, l in enumerate(label):
    temp = [0 for j in range(891)]
    temp[l[1]] = 1
    y.append(temp)
y = np.array(y)
X_temp = []
for i in range(26)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

history = LossHistory()

def build_model():
    model=Sequential()
    model.add(Masking(mask_value= 0,input_shape=(maxleng, 30)))
    model.add(LSTM(30,return_sequences=True))
    model.add(LSTM(30,return_sequences=True))
    model.add(TimeDistributed(Dense(885)))
    model.add(Activation('softmax'))
    r = RMSprop(lr=0.001)
    model.compile(optimizer=r,loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    return model

model=build_model()
model.summary()
model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=1,epochs=2000,batch_size=16, callbacks=[history])
history.loss_plot('epoch')
predict=model.predict(x_test)
print('sss')