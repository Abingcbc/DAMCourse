import numpy as np
import pandas as pd
import logging
import json
import time
import gensim.models

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
start = time.process_time()
DIM = 300

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec_model.bin", binary=True)

with open("../review_dianping_12908_split_comma.json") as review_file:
    review_data = json.load(review_file)

keys_over_threshold = list(word2vec_model.vocab.keys())
user_id = list(review_data.keys())
X = np.empty((1,DIM))

times = 0
length = len(review_data)

for i in range(length):
    if times%1000 == 0:
        logging.info("------" + str(times) + "------")
    X_i = np.zeros((1,DIM))
    count = 0
    for word in review_data[user_id[i]]:
        if word in keys_over_threshold:
            X_i += np.array(word2vec_model[word]).reshape((1,DIM))
            count += 1
    if count == 0:
        logging.warning("No word in user's review.")
    else:
        X_i = X_i / count
    X = np.concatenate([X,X_i])
    times += 1

X = np.delete(X, 0, axis=0)

feature = pd.DataFrame(data=X, index=user_id)
feature.to_csv("feature_word.csv")

cost_time = time.process_time() - start
print("Time used: " + str(cost_time))
