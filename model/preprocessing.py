
import urllib.request
import json
import re
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from typing import List
from PyKomoran import *
komoran = Komoran("EXP")
tqdm.pandas()


def cleanText(readData):
    text = re.sub(r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》\’\“\”\·\n\r\t■◇◆▶;\xa0]', '', readData).strip()
    return text


def morp(strings):
    return [w.get_first()+'다' if w.get_second() in ['VV','VA'] else w.get_first() for w in komoran.get_list(cleanText(strings)) if w.get_second() in ['NNP','NNG']]#'MAG','VA','VV','MM']]


def load_stopwords(path : str) -> List[str]:
    ff = open(path, "r") 
    data = ff.read()
    ff.close()
    return data.split("\n")


def preprocessing(stopwords_path : str):
    with urllib.request.urlopen("https://raw.githubusercontent.com/e9t/nsmc/master/synopses.json") as url:
        data = json.load(url)
    df = pd.DataFrame(data)
    df = df.dropna(subset=['synopsis', 'title_kr'])
    df['titlecontents'] = df.apply(lambda x:x['title_kr']+"\n"+x['synopsis'],axis=1)
    df['tokens'] = df['titlecontents'].progress_map(lambda x:morp(x))
    stopwords = set(load_stopwords(stopwords_path))
    df['tokens'] = df['tokens'].map(lambda x:[w for w in x if not w in stopwords])
    return df


def term_matrix():    
    stopwordsFile = '../datasets/stopwords.txt'
    df = preprocessing(stopwordsFile)
    tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                                    lowercase=False,
                                    tokenizer=None,
                                    preprocessor=None,
                                    min_df=5, 
                                    ngram_range=(1,2),
                                    smooth_idf=True,
                                    max_features=1000
                                    )
    tfidf_vector = tfidf_vectorizer.fit_transform(df['tokens'].astype(str))

    tfidf_scores = tfidf_vector.toarray().sum(axis=0)
    tfidf_idx = np.argsort(-tfidf_scores)
    tfidf_scores = tfidf_scores[tfidf_idx]
    tfidf_vocab = np.array(tfidf_vectorizer.get_feature_names_out())[tfidf_idx]

    ##TF-IDF 기준 Term-Term Matrix
    tfidf_term_term_mat = cosine_similarity(tfidf_vector.T)
    tfidf_term_term_mat = pd.DataFrame(tfidf_term_term_mat,
                                       index = tfidf_vectorizer.vocabulary_,
                                       columns = tfidf_vectorizer.vocabulary_)
    return tfidf_term_term_mat

import networkx as nx
import matplotlib.pyplot as plt
# %matplotlib inline
from fa2 import ForceAtlas2

df = term_matrix()

forceatlas2 = ForceAtlas2(outboundAttractionDistribution=True,  # Dissuade hubs
                          linLogMode=False,  # NOT IMPLEMENTED
                          adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                          edgeWeightInfluence=1.0,
                          
                          jitterTolerance=1.0,  # Tolerance
                          barnesHutOptimize=True,
                          barnesHutTheta=1.2,
                          multiThreaded=False,  # NOT IMPLEMENTED

                          # Tuning
                          scalingRatio=2.0,
                          strongGravityMode=False,
                          gravity=1.0,

                          # Log
                          verbose=True)

G = nx.random_geometric_graph(400, 0.2)
positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
nx.draw_networkx_nodes(G, positions, node_size=20, with_labels=False, node_color="blue", alpha=0.4)
nx.draw_networkx_edges(G, positions, edge_color="green", alpha=0.05)
plt.axis('off')
plt.show()