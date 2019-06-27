import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
from keras.layers import Dense, Conv2D,Flatten, Dropout, TimeDistributed, RepeatVector
from keras.optimizers import Adam
from keras.layers import TimeDistributed
from utils import distance_loss, median_absolute_deviation

import keras
import matplotlib.pyplot as plt
import utm
from sklearn.preprocessing import minmax_scale, MinMaxScaler

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


data = pd.read_csv('hwphase2data_update/new_formatted_data.csv')
X_temp = [
    np.concatenate([mr[4:9],mr[9:14],mr[14:19],mr[19:24],mr[24:29],
               mr[29:34]]) for mr in data.values
]
X_temp = np.array(X_temp)
y_temp = []
zone_number = 0
zone_letter = 0
for mr in data.values:
    lat, lon, zone_number, zone_letter = utm.from_latlon(mr[-4], mr[-5])
    y_temp.append([lat, lon])
y_temp = np.array(y_temp)
scaler = MinMaxScaler()
y_temp = scaler.fit_transform(y_temp)

traj = [0,1,2,3,4,5,6,7,8,10]
length = [22,120,28,36,39,22,83,300,272]
X = []
y = []
for t in range(len(traj)):
    temp = X_temp[(data['IMSI'] == 460012796993062.0) & (data['TrajID'] == traj[t])]
    temp_y = y_temp[(data['IMSI'] == 460012796993062.0) & (data['TrajID'] == traj[t])]
    for i in range(len(temp)-8):
        X.append(temp[i:i+8])
        y.append(temp_y[i:i+8])
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

history = LossHistory()

def build_model():
    m=Sequential()
    m.add(LSTM(150,return_sequences=True,input_shape=(8,30)))
    m.add(LSTM(150,return_sequences=True))
    m.add(LSTM(150,return_sequences=True))
    m.add(TimeDistributed(Dense(1024, activation='relu')))
    m.add(TimeDistributed(Dense(2, activation='linear')))
    r = Adam(lr=0.001)
    m.compile(optimizer=r,loss=distance_loss,metrics=['mse'])
    return m

model=build_model()
model.summary()
model.fit(X, y, epochs=10, validation_data=(X_test, y_test), batch_size=64, callbacks=[history])
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1,2))
y_test = scaler.inverse_transform(y_test.reshape(-1,2))
print(median_absolute_deviation(y_pred,y_test))