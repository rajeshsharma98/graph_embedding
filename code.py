import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

kaggle = pd.read_csv('dataset/nutrition_kaggle.csv')
kaggle_copy = kaggle.copy() # copy of original data
kaggle.drop(['Unnamed: 0','serving_size'],inplace=True,axis=1)
clmns = kaggle.columns
kaggle = kaggle.astype(float)
kaggle = (kaggle-kaggle.min())/(kaggle.max()-kaggle.min())



# Remove all unit values like sodium = 9.00 mg then we need to remove mg
# Iterate over all column and if column data type is Object then remove all string characters present in column values
for i in clmns[1:]:
    if kaggle[i].dtype == 'O':
        kaggle[i] = kaggle[i].str.replace('[a-zA-Z]', '')
    else:
        continue

'''EDA- still remaining'''
# Count null values in every column
kaggle.isnull().sum(axis = 0)
kaggle.fillna(9999,inplace=True)

'''1. Standardize the data'''
kaggle[clmns[1:]] = kaggle[clmns[1:]].apply(pd.to_numeric)
kaggle[clmns[1:]]= (kaggle[clmns[1:]]-kaggle[clmns[1:]].min())/(kaggle[clmns[1:]].max()-kaggle[clmns[1:]].min())

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

kaggle.set_index('name', inplace=True)
names = kaggle.index.to_list()

# Add weight to each edge
clmns = clmns[1:]
for i in names:
    for j in clmns:
        B[i][j]['weight'] = kaggle.loc[i,j]

'''2.  Remove Edges with weight=0 '''
edge_wgt = nx.get_edge_attributes(B,'weight')

B.remove_edges_from((edge for edge, weight in edge_wgt.items() if weight == 0))
# 8828 edges removed