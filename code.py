# Code to check if required libraries installed or not 
# if not do something

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import random
from sklearn.cluster import KMeans

# from tqdm import tqdm
# from gensim.models.word2vec import Word2Vec
warnings.filterwarnings("ignore", category=FutureWarning)

kaggle = pd.read_csv('dataset/nutrition_kaggle.csv')
kaggle_copy = kaggle.copy() # copy of original data
kaggle.drop(['Unnamed: 0','serving_size'],inplace=True,axis=1)
clmns = kaggle.columns
print('Dataset loaded\n')

# Remove all unit values like sodium = 9.00 mg then we need to remove mg
# Iterate over all column and if column data type is Object then remove all string characters present in column values
for i in clmns[1:]:
    if kaggle[i].dtype == 'O':
        kaggle[i] = kaggle[i].str.replace('[a-zA-Z]', '')
    else:
        continue
#--------------------------------------------

'''EDA- still remaining'''
# Count null values in every column
(kaggle.isnull().sum(axis = 0)).sum()
kaggle = kaggle.fillna(0)
(kaggle.isnull().sum(axis = 0)).sum()
kaggle.fillna(9999,inplace=True)

'''1. Standardize the data'''
kaggle[clmns[1:]] = kaggle[clmns[1:]].apply(pd.to_numeric)
kaggle[clmns[1:]]= (kaggle[clmns[1:]]-kaggle[clmns[1:]].min())/(kaggle[clmns[1:]].max()-kaggle[clmns[1:]].min())
kaggle[clmns[1:]] = kaggle[clmns[1:]].astype(str)
kaggle.to_csv('srtandardized_dataset.csv')

''' Graph Creation'''
# Stacking
# willnot include any column wiht null values
ing_nut = kaggle.set_index('name').stack()
# edges
edges = ing_nut.index.tolist()

# ''' Approach to create Bipartite graph directly'''
B = nx.Graph()
B.add_nodes_from(kaggle['name'].tolist(), bipartite=0)
B.add_nodes_from(kaggle.columns.tolist(), bipartite=1)
B.add_edges_from(edges)
print('Bipartite graph created\n')

kaggle.set_index('name', inplace=True)
names = kaggle.index.to_list()

# Add weight to each edge
clmns = clmns[1:]
for i in names:
    for j in clmns:
        B[i][j]['weight'] = kaggle.loc[i,j]

'''2.  Remove Edges with weight=0 '''
edge_wgt = nx.get_edge_attributes(B,'weight')
before = len(B.edges)
print('Number of edges before removal: ',before)
B.remove_edges_from((edge for edge, weight in edge_wgt.items() if weight == '0.0'))
print('Edges with weight 0 removed')
after = len(B.edges)
print('Number of edges after removal: ',after)
print('Number of edges removed: ',before-after,'\n' )

''' 2.1 '''
# count number of zeros in the dataset
zero = ((kaggle == '0.0').sum()).sum()
zero == before-after

#--------------------------------------------
# Toy Graph
m, n = 3, 4
K = nx.complete_bipartite_graph(m, n)
print('Toy graph created\n')
### To plot the toy graph- without edge weights
# pos = {}
# pos.update((i, (i - m/2, 1)) for i in range(m))
# pos.update((i, (i - m - n/2, 0)) for i in range(m, m + n))
# fig, ax = plt.subplots()
# fig.set_size_inches(15, 4)
# nx.draw(K, with_labels=True, pos=pos, node_size=300, width=0.4)
# plt.show()

# Assign random weights to each edge- weight ranges between 0 and 1
for (u, v) in K.edges():
    K.edges[u,v]['weight'] = round(random.random(),3)
print('Toy graph weights added\n')

pos = {}
pos.update((i, (i - m/2, 1)) for i in range(m))
pos.update((i, (i - m - n/2, 0)) for i in range(m, m + n))
fig, ax = plt.subplots()
fig.set_size_inches(15, 4)
labels = nx.get_edge_attributes(K,'weight')
nx.draw(K,pos)
nx.draw_networkx_edge_labels(K,  pos=pos, edge_labels=labels)
plt.show()
print('Toy graph plotted with weights\n')

#--------------------------------------------
''' Professor code run''' # helper-2.ipynb used
# // TODO Need to add weights to walker
import networkx as nx
import random
import numpy as np
from typing import List
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec

class DeepWalk:
    def __init__(self, window_size: int, embedding_size: int, walk_length: int, walks_per_node: int):
        """
        ATTRIBUTES:  
        window_size: window size for the Word2Vec model
        embedding_size: size of the final embedding
        walk_length: length of the walk
        walks_per_node: number of walks per node
        """
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.walk_length = walk_length
        self.walk_per_node = walks_per_node

    def generate_walks(self, g: nx.Graph, use_probabilities: bool = False) -> List[List[str]]:
        """
        ATTRIBUTES:
        g: Graph
        use_probabilities: use edge weights as probabilioties for random walk
        """
        walks = []
        for _ in range(self.walk_per_node):
            random_nodes = list(g.nodes)
            random.shuffle(random_nodes)
            for node in tqdm(random_nodes):  # tqdm generated progress bar- will be useful in the original graph as it is big
              walk = [node] # node =start
              for i in range(self.walk_length):
                  neighbours = g.neighbors(walk[i])
                  neighs = list(neighbours)
                  if use_probabilities:
                      probabilities = [g.get_edge_data(walk[i], neig)["weight"] for neig in neighs]
                      sum_probabilities = sum(probabilities)
                      probabilities = list(map(lambda t: t / sum_probabilities, probabilities))
                      p = np.random.choice(neighs, p=probabilities)
                  else:
                      p = random.choice(neighs)
                  walk.append(p)
              walks.append(walk)
        return walks

# //TODO : insert following function inside above class
def get_embedding(G, walks, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
    kwargs["sentences"] = walks
    kwargs["min_count"] = kwargs.get("min_count", 0)
    kwargs["size"] = embed_size
    kwargs["sg"] = 1  # skip gram
    kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
    kwargs["workers"] = workers
    kwargs["window"] = window_size
    kwargs["iter"] = iter

    print("Learning embedding vectors...")
    model = Word2Vec(**kwargs)
    print("Learning embedding vectors done!")

    embeddings = {}
    for word in G.nodes():
        embeddings[str(word)] = model.wv[str(word)] # Need to convert node value to string else int not iterable
    return embeddings


# Try the embedding in toy graph
p = DeepWalk(2,2,2,2)
wal = p.generate_walks(K,use_probabilities=True)

# need to convert walk values to string datatype as by default node value are of inte type and therefore the nested 
# list items also
z1 = []
for i in wal:
  z2 = []
  for j in i:
    j = str(j)
    z2.append(j)
  z1.append(z2)

# get_embedding(K,z1)

''' //TODO plot the embeddings of toy graph in 2D space- embedding_size=2'''

# wlak_length = 10 and 12

#--------------------------------------------
''' //TODO to decide cluster'''

df = pd.DataFrame.from_dict(get_embedding(K,z1), orient='index')

kmeans = KMeans(n_clusters=2, random_state=0).fit(df)

kmeans.labels_

# Embedding on original graph

