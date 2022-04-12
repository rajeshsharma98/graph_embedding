# Ingredient recommendation  
Steps:  
1. // [x] Standardize the data such that every column range between 0 and 1   
      [ ] Check if standardizattion was implemented  
      [x] Store the standardize data as a seperate file    

2. // [x] Remove edge with weight = 0   
      - [x] Check number of edges removed == number of zeros in the dataset  

3. // [ ] baseline model (base_model.py)  
      - [x] Check for null values   
      - [x] Standardize data  
      - [x] PCA implementation  
      - [ ] Run ML algorothim  
      - [ ] Get the accuracy  

4. // [ ] Implement Random walk - edge weight is the probability of that path(or edge).  
      [x] Create toy bipartite graph   
      [x] Verify random walk(without weights) and get the embeddings of toy graph   
      [x] Tweak the algorithm to cosider weight of edges as probabilities  
      [x] Implement random walk on toy graph using weights 
      [ ] Verify random walk and get the embeddings of toy graph    
      [ ] Implement random walk on main graph using weights  
      [ ] Save the embedding  

5. // [ ] Clustering  
      [ ] For default paramters  
      [ ] Decide number of clusters  
6. // [ ] KNN (K-Nearest Neighbor)    

Code files:  
1. base_model.py - Baseline model implementation  
2. embed.py - graph embedding - deep walk for now  
3. code.py - data loading, standardization and embedding implementation   

Meeting (Apr 4):  
-> Random walk must be of even length, such that after each walk it should come to right part (that represent ingredient) of bipartite graph  
-> Base line model  - Consider whole dataset as matrix (not a graph), use PCA and then apply Clustering or KNN  



Reference:  
GitHub docs- https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax    