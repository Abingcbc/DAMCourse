import numpy as np
import pandas as pd
import os.path
import re
from clarifai.rest import ClarifaiApp
import json

user_avatar_dic = {}

def getLabels(filename, user_id):
    app = ClarifaiApp(api_key="b7f6f2f48d5c4618bf2980a539a3a554")
    model = app.public_models.general_model
    response = model.predict_by_filename(filename)
    response = response['outputs'][0]['data']['concepts']
    labels = []
    for dic in response:
        labels.append((dic['name'],dic['value']))
    user_avatar_dic[user_id] = labels

def eachFile(filepath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        getLabels(child,allDir[:allDir.index("_")])


# 遍历出结果 返回文件的名字
# def readFile(filenames):
#     fopen = open(filenames, 'r')  # r 代表read
#     fileread = fopen.read()
#     fopen.close()
#     t = re.search(r'clearSpitValve', fileread)
#     # if t:
#         #             print "匹配到的文件是:"+filenames
#         # arr.append(filenames)

eachFile("/Users/cbc/Desktop/homework/DataMining/hw2/dianping_avatar")
