''' 3.1. '''

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.decomposition import PCA


kaggle = pd.read_csv('dataset/nutrition_kaggle.csv')
kaggle_copy = kaggle.copy() # copy of original data
kaggle.drop(['Unnamed: 0','serving_size'],inplace=True,axis=1)
clmns = kaggle.columns
# print('Dataset loaded\n')

# Remove all unit values like sodium = 9.00 mg then we need to remove mg
# Iterate over all column and if column data type is Object then remove all string characters present in column values
for i in clmns[1:]:
    if kaggle[i].dtype == 'O':
        kaggle[i] = kaggle[i].str.replace('[a-zA-Z]', '')
    else:
        continue

''' //TODO: EDA Remaining'''
# Count null values in every column
''' 3.2. Check for null values '''
(kaggle.isnull().sum(axis = 0)).sum()
kaggle = kaggle.fillna(0)
print((kaggle.isnull().sum(axis = 0)).sum())
kaggle.fillna(9999,inplace=True)

''' Correlation '''
kaggle[clmns[1:]] = kaggle[clmns[1:]].apply(pd.to_numeric)
corr = kaggle.corr()
# sns.heatmap(corr, annot=True)
# plt.show()
# Plotting not possible due to large feature set

'''//TODO:  Get pairs above specified correlation value'''
# Plot columns whose correlation greteer than specified range
kot = corr[corr>=.8]
# plt.figure(figsize=(12,8))  # uncomment this and below line to plot the matrix
# sns.heatmap(kot, cmap="Greens")

''' //TODO: Get pair name for above specified range'''

'''3.3. - 3.4. Standardization & PCA'''
# kaggle[clmns[1:]]= (kaggle[clmns[1:]]-kaggle[clmns[1:]].min())/(kaggle[clmns[1:]].max()-kaggle[clmns[1:]].min())
X = kaggle[clmns[1:]]
X_normalized =(X - X.mean()) / X.std()
y = kaggle.name


pca = PCA(n_components = 0.95)
pca.fit(X)
X_reduced = pca.transform(X)

''' //TODO: Split data into train and test'''

''' //TODO: 3.5. Apply required ML algorithm'''

''' //TODO: 3.6. Get accuracy'''
