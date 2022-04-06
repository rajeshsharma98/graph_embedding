# graph_embedding  
Steps:  
1. ~~Standardize the data such that every column range between 0 and 1 ~~
2. ~~Remove edge with weight = 0 ~~
3. Implement Random walk   
4. Clustering or KNN (K-Nearest Neighbor)   

Meeting (Apr 4):  
-> Random walk must be of even lenght, such that after each walk it should come to right part (that represent ingredient) of bipartite graph  
-> Base line model  - Consider whole dataset as matrix (not a graph), use PCA and then apply Clustering or KNN  