# Clustering by fast search and find of density peaks

## 运行环境

* Python 3.7
* Numpy
* Matplotlib

## 运行方式

* 对`Aggregation.txt ` 进行聚类(需要将`Aggregation.txt`放在同一文件夹内) 

  ```shell
  python density_peak.py
  ```

* 对自定义数据进行聚类

  ```python
  predict(X, dc, NUM_OF_CENTER, plot=False, is_hola=False)
  ```

  * X: 训练样本
  * NUM_OF_CENTER : 聚类中心个数
  * plot : 是否绘图
  * is_hola : 是否寻找异常点

* 选取dc，返回使得范围内的平均样本数为总样本数1%～2%的dc

  ```python
  chooseDc(candidate, X)
  ```

  * candidate : dc候选范围
  * X : 训练样本



