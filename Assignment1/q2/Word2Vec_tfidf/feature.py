import numpy as np
import pandas as pd
import logging
import json
import time
import gensim.models
from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
start = time.process_time()
DEM = 300

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec_model.bin", binary=True)
tf_idf_model = gensim.models.TfidfModel.load("tf_idf_model.bin")

with open("../review_dianping_12908_split_comma.json") as review_file:
    review_data = json.load(review_file)

dictionary = corpora.Dictionary.load("tf_idf_dictionary.bin")
corpus = corpora.MmCorpus("corpus.bin")

keys_index_order = list(dictionary.token2id.keys())
keys_over_threshold = list(word2vec_model.vocab.keys())
user_id = list(review_data.keys())
X = np.empty((1,DEM))

times = 0
length = len(review_data)

for i in range(length):
    if times%1000 == 0:
        logging.info("------" + str(times) + "------")
    X_i = np.zeros((1,DEM))
    count = 0
    tf_idf_cur = tf_idf_model[corpus[i]]
    for item in tf_idf_cur:
        word = keys_index_order[item[0]]
        if word in keys_over_threshold:
            X_i += item[1] * np.array(word2vec_model[word]).reshape((1,DEM))
            count += 1
    if count == 0:
        logging.warning("No word in user's review.")
    else:
        X_i = X_i / count
    X = np.concatenate([X,X_i])
    times += 1

X = np.delete(X, 0, axis=0)

feature = pd.DataFrame(data=X, index=user_id)
feature.to_csv("feature_word_tfidf.csv")

cost_time = time.process_time() - start
print("Time used: " + str(cost_time))
