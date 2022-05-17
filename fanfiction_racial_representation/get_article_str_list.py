import os
import whatlies
from whatlies import Embedding, EmbeddingSet
article_list = []
filename_list = []
for root,directories,files in os.walk("output",topdown=False) :
    for name in files :
        #print(os.path.join(root,name))
        filename_list.append(os.path.join(root, name))
    for name in directories :
        print(os.path.join(root,name))
for filename in filename_list:
    fd=open(filename)
    article_list.append(fd.read())

pass

from gensim.utils import simple_preprocess
preprocessed_docs = []
for doc in article_list:
    preprocessed_docs.append(simple_preprocess(doc, min_len=1, max_len=20))

from gensim.models import Word2Vec

w2v_model1 = Word2Vec(sentences=preprocessed_docs,
                      vector_size=50,
                      window=5,
                      min_count=5,
                      workers=3,
                      sg=1,
                      hs=0,
                      negative=5,
                      epochs=10)

emb_w2v = EmbeddingSet.from_names_X(names=w2v_model1.wv.index_to_key,
                                    X=w2v_model1.wv.vectors)
(emb_w2v["society", "nation", "individual", "exclusive"]
 .plot(x_axis="society",
       y_axis="individual",
       axis_metric="cosine_similarity"))
pass
