from .random_walk import random_walk
import random

def walker(adj_list, walks, walk_len, walker):
    result = []
    for i in range(walks):
        random.shuffle(adj_list)
        for v in range(len(adj_list)):
            if(walker == 'random_walk'):
                result.append(random_walk(adj_list, walk_len, v))
    return result