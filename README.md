# Ingredient recommendation  
Steps:  
1. - [x] Standardize the data such that every column range between 0 and 1 
2. - [x] Remove edge with weight = 0 
      - [x] Check number of edges removed == number of zeros in the dataset  
4. - [ ] Implement Random walk   
5. - [ ] Clustering or KNN (K-Nearest Neighbor)   

Meeting (Apr 4):  
-> Random walk must be of even lenght, such that after each walk it should come to right part (that represent ingredient) of bipartite graph  
-> Base line model  - Consider whole dataset as matrix (not a graph), use PCA and then apply Clustering or KNN  