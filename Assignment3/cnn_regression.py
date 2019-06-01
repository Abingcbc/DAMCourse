import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import KFold

data = pd.read_csv('/Users/cbc/Project/Python/DAMCourse/Assignment3/data/train_2g.csv')

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
print(X[0])

result_mse = []
result_mae = []
for train_index, val_index in floder.split(X, y):
    model = build_model()
    model.fit(X[train_index], y[train_index], epochs=10, batch_size=1)
    val_mse, val_mae = model.evaluate(X[val_index], y[val_index])
    result_mse.append(val_mse)
    result_mae.append(val_mae)

print(np.mean(result_mse))
print(np.mean(result_mae))
