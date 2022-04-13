import random

def random_walk(adj_list, walk_len, start_node):
    walk = [str(start_node)]
    while len(walk) < walk_len:
        cur = int(walk[-1])
        neighbours = adj_list[cur]
        if len(neighbours) > 0:
            walk.append(random.choice(neighbours))
        else:
            break
    return walk