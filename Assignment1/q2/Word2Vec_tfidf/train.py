from gensim.models import TfidfModel
from gensim.models import word2vec
from gensim import corpora
import logging
import json
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with open("../review_dianping_12908_split_comma.json") as review_file:
    review_data = json.load(review_file)

sentences = list(review_data.values())
'''
There is null in the raw data, 
but I think the reason causes null can be a feature of the review.
So I didn't remove null.

Position of null:
word2vec key: 65
tf-idf dictionary: 0
'''
start = time.time()

word2vec_model = word2vec.Word2Vec(sentences, size=300,
                                   workers=5, min_count=10)
word2vec_model.wv.save_word2vec_format("word2vec_model.bin", binary=True)

dictionary = corpora.Dictionary(sentences) # 89752
dictionary.save("tf_idf_dictionary.bin")

corpus = [dictionary.doc2bow(text) for text in sentences]
corpora.MmCorpus.serialize("corpus.bin", corpus)

tf_idf_model = TfidfModel(corpus)
tf_idf_model.save("tf_idf_model.bin")

print("Time Used: " + str(time.time() - start))