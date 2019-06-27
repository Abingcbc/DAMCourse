# 数据挖掘Assignment3

## 数据预处理

首先，将所有缺失基站的信息用该mr数据的第一条基站信息填充。接着，分别对每个基站信息相同属性的5列进行归一化，更加严谨。刚开始，分别对训练集和测试集分开归一化，得到的预测结果很差，明显不符合训练时的表现。将训练集和测试集合并归一化后效果明显正常。CNN按照要求进行了处理，LSTM以6为步长进行了划分。

## CNN回归

### Code

```python
def build_model():
    m = Sequential()
    m.add(Conv2D(32, kernel_size=2, activation='relu', input_shape=(5, 6, 1)))
    m.add(Conv2D(16, kernel_size=2, activation='relu'))
    m.add(Flatten())
    m.add(Dense(1024, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(512, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(512, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(256, activation='relu'))
    m.add(Dense(2))
    r = optimizers.Adam(lr=0.001,decay=0.00001)
    m.compile(optimizer=r, loss=distance_loss, metrics=['mse'])

    return m
```

### Deviation

|     平均误差      |      中位误差      |      90%误差      |
| :---------------: | :----------------: | :---------------: |
| 31.38238199797868 | 17.975134224167107 | 52.15832459106061 |

### CDF

<img src='/Users/cbc/Project/Python/DAMCourse/Assignment3/cnn_regression_cdf.png' width='300px'>

### Map

<img src='/Users/cbc/Desktop/images/屏幕快照 2019-06-22 08.59.36.png' width='300px'>



## CNN分类

### Code

```python
def build_model():
    m = Sequential()
    m.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(5, 6, 1)))
    m.add(Conv2D(16, kernel_size=3, activation='relu'))
    m.add(Flatten())
    m.add(Dense(1024, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(512, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(512, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(891, activation='softmax'))
    r = optimizers.Adam(lr=0.001, decay=0.00001)
    m.compile(optimizer=r, loss='categorical_crossentropy', metrics=['accuracy'])
    return m
```

### Deviation

|     平均误差      |      中位误差      |      90%误差       |
| :---------------: | :----------------: | :----------------: |
| 59.77612740459058 | 24.066227454845617 | 108.14002709652033 |

### CDF

<img src='/Users/cbc/Project/Python/DAMCourse/Assignment3/cnn_classify_cdf.png' width='300px'>

### Map

<img src='/Users/cbc/Desktop/images/屏幕快照 2019-06-23 10.23.56.png' width='300px'>

## LSTM回归

### Code

### Deviation

IMSI = 460012796993062

|      平均误差      |     中位误差      |     90%误差      |
| :----------------: | :---------------: | :--------------: |
| 15.620514905833225 | 8.888001236191048 | 24.7304063586534 |

IMSI = 460016291512177

|     平均误差      |      中位误差      |      90%误差      |
| :---------------: | :----------------: | :---------------: |
| 7.595752656386923 | 4.3100168539066175 | 13.09191939509659 |

IMSI = 460011670515939

|      平均误差      |     中位误差      |      90%误差      |
| :----------------: | :---------------: | :---------------: |
| 11.498884483041163 | 8.862193073190014 | 20.36031539399085 |

IMSI = 460091042103139

|     平均误差     |     中位误差      |      90%误差       |
| :--------------: | :---------------: | :----------------: |
| 9.01460344532173 | 4.591861951287191 | 16.067524848618344 |

### CDF

IMSI = 460012796993062

<img src='/Users/cbc/Project/Python/DAMCourse/Assignment3/lstm_regression_1_cdf.png' width='300px'>

IMSI = 460016291512177

<img src='/Users/cbc/Project/Python/DAMCourse/Assignment3/lstm_regression_2_cdf.png' width='300px'>

IMSI = 460011670515939

<img src='/Users/cbc/Project/Python/DAMCourse/Assignment3/lstm_regression_3_cdf.png' width='300px'>

IMSI = 460091042103139

<img src="/Users/cbc/Project/Python/DAMCourse/Assignment3/lstm_regression_4_cdf.png" width='300px'>



### Map

<img src='/Users/cbc/Desktop/images/屏幕快照 2019-06-27 01.10.56.png' width='300px'>

## LSTM分类

### Code

```python
def build_model():
    model=Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(maxleng,30)))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(TimeDistributed(Dense(891)))
    model.add(Activation('softmax'))
    r = Adam(lr=0.001)
    model.compile(optimizer=r,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
```

### Deviation

|      平均误差      |      中位误差      |      90%误差      |
| :----------------: | :----------------: | :---------------: |
| 37.304825766308305 | 22.080358749456323 | 46.94440747830585 |

### CDF

<img src='/Users/cbc/Project/Python/DAMCourse/Assignment3/lstm_classify_cdf.png' width='300px'>

### Map

<img src='/Users/cbc/Desktop/images/屏幕快照 2019-06-27 17.37.43.png' width='300px'>

## CNN-LSTM分类

### Code

```python
def build_model_cnn():
    m = Sequential()
    m.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(5, 6, 1)))
    m.add(Conv2D(64, kernel_size=3, activation='relu'))
    m.add(Flatten())
    m.add(Dense(1024, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(512, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(512, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(891, activation='softmax'))
    r = optimizers.Adam(lr=0.001)
    m.compile(optimizer=r, loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_model_lstm():
    model=Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(maxleng,891)))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dense(1024))
    model.add(TimeDistributed(Dense(891)))
    model.add(Activation('softmax'))
    r = RMSprop(lr=0.001)
    model.compile(optimizer=r,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
```

### Deviation

|      平均误差      |     中位误差      |      90%误差      |
| :----------------: | :---------------: | :---------------: |
| 24.589870115229196 | 20.93615175580477 | 32.37773792102032 |

### CDF

<img src='/Users/cbc/Project/Python/DAMCourse/Assignment3/cnn_lstm_cdf.png' width='300px'>

### Map

<img src='/Users/cbc/Desktop/images/屏幕快照 2019-06-27 17.00.33.png' width='300px'>

## 自编码器-LSTM分类

### Code

```python
def build_model_ae():
    m = Sequential()
    m.add(LSTM(64, return_sequences=True, input_shape=(6,30)))
    m.add(LSTM(32, return_sequences=True))
    m.add(TimeDistributed(Dense(30)))
    r = Adam(lr=0.001)
    m.compile(optimizer=r, loss='mse', metrics=['mse'])
    return m

def build_model_lstm():
    m = Sequential()
    m.add(LSTM(32, return_sequences=True, input_shape=(6, 64)))
    m.add(LSTM(64, return_sequences=True))
    m.add(TimeDistributed(Dense(1024, activation='relu')))
    m.add(TimeDistributed(Dense(1024, activation='relu')))
    m.add(TimeDistributed(Dense(891, activation='softmax')))
    r = Adam(lr=0.001)
    m.compile(optimizer=r, loss='categorical_crossentropy', metrics=['accuracy'])
    return m
```

### Deviation

|     平均误差      |      中位误差      |      90%误差       |
| :---------------: | :----------------: | :----------------: |
| 34.62295285940166 | 21.423317583139863 | 39.038467972522646 |

### CDF

<img src='/Users/cbc/Project/Python/DAMCourse/Assignment3/antoencoder_cdf.png' width='300px'>

### Map

<img src='/Users/cbc/Desktop/images/屏幕快照 2019-06-27 17.44.02.png' width='300px'>
