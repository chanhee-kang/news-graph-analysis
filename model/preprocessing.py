
import urllib.request
import json
import re
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from networkx.readwrite import json_graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from fa2 import ForceAtlas2
from tqdm import tqdm
from typing import List
from PyKomoran import *
komoran = Komoran("EXP")
tqdm.pandas()


def cleanText(readData):
    text = re.sub(r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》\’\“\”\·\n\r\t■◇◆▶;\xa0]', '', readData).strip()
    return text


def morp(strings : str):
    return [w.get_first()+'다' if w.get_second() in ['VV','VA'] else w.get_first() for w in komoran.get_list(cleanText(strings)) if w.get_second() in ['NNP','NNG']]#'MAG','VA','VV','MM']]


def load_stopwords(path : str) -> List[str]:
    ff = open(path, "r") 
    data = ff.read()
    ff.close()
    return data.split("\n")


def tokenize(filepath : str, openwith : str) -> pd.DataFrame :    
    if openwith == "url" :
        with urllib.request.urlopen(filepath) as url:
            data = json.load(url)
            df = pd.DataFrame(data)
    elif openwith == "csv" :
        df = pd.read_csv(filepath)
    
    df = df.dropna(subset=['synopsis', 'title_kr'])
    df['titlecontents'] = df.apply(lambda x:x['title_kr']+"\n"+x['synopsis'],axis=1)
    df['tokens'] = df['titlecontents'].progress_map(lambda x:morp(x))
    stopwords = set(load_stopwords(path = '../datasets/stopwords.txt'))
    df['tokens'] = df['tokens'].map(lambda x:[w for w in x if not w in stopwords])
    return df


def term_matrix(filepath, openwith) -> pd.DataFrame :     
    df = tokenize(filepath, openwith = openwith)
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

    ##TF-IDF x cosine similarity
    tfidf_term_term_mat = cosine_similarity(tfidf_vector.T)
    tfidf_term_term_mat = pd.DataFrame(tfidf_term_term_mat,
                                       index = tfidf_vectorizer.vocabulary_,
                                       columns = tfidf_vectorizer.vocabulary_)
    return tfidf_term_term_mat

if __name__ == "__main__":
    filepath = "https://raw.githubusercontent.com/e9t/nsmc/master/synopses.json"
    dummy = term_matrix(filepath, openwith = "url")
    with open('../datasets/matrix/term_matrix_test.pickle', 'wb') as f:
        pickle.dump(dummy, f, pickle.HIGHEST_PROTOCOL)



df = term_matrix()
graph = nx.from_numpy_matrix(df.values)

forceatlas2 = ForceAtlas2(outboundAttractionDistribution=True,  # Dissuade hubs
                          linLogMode=False,  # NOT IMPLEMENTED
                          adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                          edgeWeightInfluence=1.0,
                          
                          jitterTolerance=5.0,  # Tolerance
                          barnesHutOptimize=True,
                          barnesHutTheta=1.2,
                          multiThreaded=False,  # NOT IMPLEMENTED

                          # Tuning
                          scalingRatio=2.0,
                          strongGravityMode=False,
                          gravity=1.0,

                          # Log
                          verbose=True)
positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
nx.draw_networkx_nodes(G, positions, node_size=20, label=True, node_color="blue", alpha=0.4)
nx.draw_networkx_edges(G, positions, edge_color="green", alpha=0.05)

G = graph
# largest connected component
components = nx.connected_components(G)
largest_component = max(components, key=len)
H = G.subgraph(largest_component)

# compute centrality
centrality = nx.betweenness_centrality(H, k=10, endpoints=True)

# compute community structure
lpc = nx.community.label_propagation_communities(H)
community_index = {n: i for i, com in enumerate(lpc) for n in com}

###레이아웃별로 뭐가 이쁠지 테스트 해보기!
layouts = {'spring': nx.spring_layout(G), 
           'spectral':nx.spectral_layout(G), 
           'shell':nx.shell_layout(G), 
           'circular':nx.circular_layout(G),
           'kamada_kawai':nx.kamada_kawai_layout(G), 
           'random':nx.random_layout(G)
          }


#### draw graph ####
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(H, k=0.15, seed=4572321)
node_color = [community_index[n] for n in H]
node_size = [v * 20000 for v in centrality.values()]


nx.draw_networkx(
    H,
    pos=pos,
    with_labels=False,
    node_color=node_color,
    node_size=node_size,
    edge_color="gainsboro",
    alpha=0.4,
)

# Title/legend
font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title("테스트!", font)
font["color"] = "r" # Change font color for legend

ax.text(
    0.80,
    0.10,
    "node color = community structure",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = betweeness centrality",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)

# Resize figure for label readibility
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.savefig("../figure/savefig_default.png")

json_graph.node_link_data(H)
