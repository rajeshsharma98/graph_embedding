from .random_walk import random_walk
import random

def walker(adj_list, walks, walk_len, walker):
    result = []
    for i in range(walks):
        for v in adj_list.keys():
            if(walker == 'random_walk'):
                result.append(random_walk(adj_list, walk_len, v))
            else:
                raise Exception("Invalid walker")
    return result