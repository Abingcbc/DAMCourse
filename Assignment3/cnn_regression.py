import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
import utm
from utils import distance_loss, median_absolute_deviation

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


data = pd.read_csv('hwphase2data_update/new_formatted_data2.csv')

X = [
    np.array([[mr[4:9],
              mr[9:14],
              mr[14:19],
              mr[19:24],
              mr[24:29],
              mr[29:34]]]).T for mr in data.values
]
X = np.array(X)
y = []
zone_number = 0
zone_letter = 0
for mr in data.values:
    lat, lon, zone_number, zone_letter = utm.from_latlon(mr[-4], mr[-5])
    y.append([lat, lon])
scaler = MinMaxScaler()
y = np.array(y)
scaler.fit(y)
y = scaler.transform(y)

def build_model():
    m = Sequential()
    m.add(Conv2D(32, kernel_size=2, activation='relu', input_shape=(5, 6, 1)))
    m.add(Conv2D(16, kernel_size=2, activation='relu'))
    m.add(BatchNormalization())
    m.add(Flatten())
    m.add(Dense(1024, activation='relu'))
    m.add(Dense(512, activation='relu'))
    m.add(Dense(64, activation='relu'))
    m.add(Dense(2))
    r = optimizers.Adam(lr=0.001,decay=0.00001)
    m.compile(optimizer=r, loss=distance_loss, metrics=['mse'])

    return m


history = LossHistory()
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = build_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=64, callbacks=[history])
history.loss_plot('epoch')
print(median_absolute_deviation(model.predict(X_test),y_test))
data_test = pd.read_csv('hwphase2data_update/new_formatted_.csv')
X_test = [
    np.array([[mr[4:9],
              mr[9:14],
              mr[14:19],
              mr[19:24],
              mr[24:29],
              mr[29:34]]]).T for mr in data_test.values
]
X_test = np.array(X_test)
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(-1,2)
y_pred = scaler.inverse_transform(y_pred)
y_final = [[utm.to_latlon(i[0], i[1], zone_number=zone_number, zone_letter=zone_letter)[1],
            utm.to_latlon(i[0], i[1], zone_number=zone_number, zone_letter=zone_letter)[0]] for i in y_pred]
y_final = np.array(y_final)
df_pred = pd.DataFrame(data={'Longitude':y_final[:,0], 'Latitude':y_final[:,1]})
df_pred.to_csv('hwphase2data_update/pred1.csv', index=False)