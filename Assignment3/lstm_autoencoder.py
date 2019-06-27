import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
from keras.layers import Dense, Input,Flatten, Dropout, TimeDistributed, RepeatVector
from keras.optimizers import Adam
from keras import backend as K
import keras
import matplotlib.pyplot as plt
import utm
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from keras.utils import to_categorical

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


def distance_loss(y_pred, y_true):
    return K.sqrt(K.mean(K.sum(K.square(y_pred-y_true), axis=-1)))

def median_absolute_deviation(y_pred, y_true):
    deviation = np.abs(y_pred-y_true)
    return np.median(deviation, axis=0)


data = pd.read_csv('hwphase2data_update/new_formatted_data2.csv')
label = pd.read_csv('hwphase2data_update/label1.csv')['label']
X_temp = [
    np.concatenate([mr[4:9],mr[9:14],mr[14:19],mr[19:24],mr[24:29],
               mr[29:34]]) for mr in data.values
]
X_temp = np.array(X_temp)
y_temp = label.values

traj = [0,1,2,3,4,5,6,7,8,10]
length = [22,120,28,36,39,22,83,300,272]
X = []
y = []
for t in range(72):
    temp = X_temp[data['TrajID'] == t]
    temp_y = y_temp[data['TrajID'] == t]
    for i in range(len(temp)-8):
        X.append(temp[i:i+8])
        y.append(temp_y[i:i+8])
X = np.array(X)
y = np.array(y)
y = to_categorical(y, 885)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
history = LossHistory()

def build_model_ae():
    m = Sequential()
    m.add(LSTM(64, return_sequences=True, input_shape=(8,30)))
    m.add(LSTM(32, return_sequences=True))
    m.add(TimeDistributed(Dense(30)))
    r = Adam(lr=0.001)
    m.compile(optimizer=r, loss='mse', metrics=['mse'])
    return m

def build_model_lstm():
    m = Sequential()
    m.add(LSTM(32, return_sequences=True, input_shape=(8, 64)))
    m.add(LSTM(64, return_sequences=True))
    m.add(TimeDistributed(Dense(1024, activation='relu')))
    m.add(TimeDistributed(Dense(1024, activation='relu')))
    m.add(TimeDistributed(Dense(885, activation='softmax')))
    r = Adam(lr=0.001)
    m.compile(optimizer=r, loss='categorical_crossentropy', metrics=['accuracy'])
    return m

model1 = build_model_ae()
model1.fit(X_train, X_train, epochs=10, batch_size=32)
model1 = Model(inputs=model1.inputs, outputs=model1.layers[0].output)
model1.summary()
X_train_vec = model1.predict(X_train)
X_test_vec = model1.predict(X_test)
model2 = build_model_lstm()
model2.fit(X_train_vec, y_train, validation_data=(X_test_vec, y_test),epochs=100, batch_size=64)
model2.summary()
# print(model2.predict(X_test))
print(model2.evaluate(X_test_vec, y_test))
