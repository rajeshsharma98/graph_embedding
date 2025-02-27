# Graph Embedding    
### This repository is sub-part of [Unique Meal-plan Prepration](https://github.com/rajeshsharma98/Unique_Meal_plan)    

We represented our dataset using graphs.   
**Bipartite graph** best represented our dataset with row names (ingredients) as one set of nodes and column names (nutritional values) as the other sets of nodes.     
**Bidirectional edges** imply the ingredient contains a particular nutritional value.    
**Edge weights** in the bipartite signify their respective normalized nutritional values.   
To segregate the nodes of the graph into two sets we used DFS node coloring algorithm.  

### Why this embeddding method not conventional methods ?    
Once we had the graph created, we started with the Embedding phase. Here we represented our graph in a lower-dimensional space. Our graph being bipartite, it only had the depth for a traversal, no width. So, the best embedding as per this analysis was using Deep-Walk. But this bipartite graph is not of the conventional form, our embedding had to be dependent on the edge list values (the values computed after normalization). To counter this, we decided to use this edge list values as the bias while making the decision for the walk. It is like a deep walk, but we considered edge weights as a probability to select the next node for the walk while producing random walks. Hence the greater the nutritional value in the ingredient, the greater the chance to visit the nutrient node while performing the walk. Biases in the random walks depend on the nutritional value in the ingredient. The walk always starts from the ingredients set of nodes and ends at the same set as our primary focus is to cluster the ingredients while considering their nutritional values as features. 
