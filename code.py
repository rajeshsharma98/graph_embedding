# Code to check if required libraries installed or not 
# if not do something

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings
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
# To plot the toy graph
pos = {}
pos.update((i, (i - m/2, 1)) for i in range(m))
pos.update((i, (i - m - n/2, 0)) for i in range(m, m + n))
fig, ax = plt.subplots()
fig.set_size_inches(15, 4)
nx.draw(K, with_labels=True, pos=pos, node_size=300, width=0.4)
plt.show()

#--------------------------------------------
''' Professor code run'''
# // TODO Need to add weights to walker
def deepwalk_walks(G, num_walks, walk_length,):
        nodes = G.nodes()
        walks = []
        for _ in range(num_walks):
            for v in nodes:
                walk = [v]
                while len(walk) < walk_length:
                    cur = walk[-1]
                    cur_nbrs = list(G.neighbors(cur))
                    if len(cur_nbrs) > 0:
                        walk.append(random.choice(cur_nbrs))
                    else:
                        break
                walks.append(walk)
        return walks

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


z = deepwalk_walks(K,2,3)
# need to convert walk values to string datatype as by default node value are of inte type and therefore the nested list items also
z1 = []
for i in z:
  z2 = []
  for j in i:
    j = str(j)
    z2.append(j)
  z1.append(z2)

get_embedding(K,z1)
''' //TODO plot the embeddings of toy graph in 2D space'''

