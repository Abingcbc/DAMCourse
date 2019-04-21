# 大众点评评论用户聚类

陈柄畅 1753837

## 运行环境

* Python 3.7
* Gensim
* Numpy
* Pandas
* Scikit-learn
* Matplotlib
* q1 (第一题中的文件需放在同目录下q1文件夹中)

## 运行方式

* 训练Word2Vec模型

  ```shell
  python train.py
  ```

* 提取特征，避免每次运行重复计算

  ```
  python feature.py
  ```

* 进行比较

  ```
  python cluster.py
  ```