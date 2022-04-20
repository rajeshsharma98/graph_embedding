from .random_walk import random_walk
from .baised_walk import baised_walk
from utils import fetch_bipartite_graph_node

import random

def walker(adj_list, walks, walk_len, walker):
    result = []
    for i in range(walks):
        if(walker == 'random_walk'):
            for v in adj_list.keys():
                result.append(random_walk(adj_list, walk_len, v))
        elif walker == 'baised_walk':
            # create bipartie graph
            s1, s2 = fetch_bipartite_graph_node(adj_list)
            nodes = []
            if len(s1) > len(s2):
                nodes = s2
            else:
                nodes = s1

            random.shuffle(nodes)
            for node in nodes:
                result.append(baised_walk(adj_list, walk_len, node))
        else:
            raise Exception("Invalid walker")
    return result
