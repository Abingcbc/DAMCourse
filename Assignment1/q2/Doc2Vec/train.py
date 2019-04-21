from gensim.models import doc2vec
import gensim
import logging
import json
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with open("../review_dianping_12908_split_comma.json") as review_file:
    review_data = json.load(review_file)

sentences = list(review_data.values())

start = time.time()
X_train = [doc2vec.TaggedDocument(sentence, [i]) for i, sentence in enumerate(sentences)]
model = gensim.models.Doc2Vec(X_train, vector_size=100, window=2, min_count=3)
model.train(X_train,total_examples=model.corpus_count, epochs=1000)

model.save("doc2vec_model.bin")

print("Time Used: " + str(time.time() - start))