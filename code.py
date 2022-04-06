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

# '''1. Standardize the data'''
kaggle[clmns[1:]] = kaggle[clmns[1:]].apply(pd.to_numeric)
kaggle[clmns[1:]]= (kaggle[clmns[1:]]-kaggle[clmns[1:]].min())/(kaggle[clmns[1:]].max()-kaggle[clmns[1:]].min())
kaggle[clmns[1:]] = kaggle[clmns[1:]].astype(str)

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