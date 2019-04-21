import gensim
import json
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.Doc2Vec.load('doc2vec_model.bin')
with open("../review_dianping_12908_split_comma.json") as review_file:
    review_data = json.load(review_file)
X = []
count = 0
for ID, sentence in review_data.items():
    if count%1000 == 0:
        logging.info("feature " + str(count) + " prepared")
    vector = model.infer_vector(sentence)
    X.append(vector)
    count += 1
logging.info("feature all prepared")
df = pd.DataFrame(X, index=list(review_data.keys()))
df.to_csv("feature_doc2vec.csv")
