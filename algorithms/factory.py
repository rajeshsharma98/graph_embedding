from .deep_walk import deep_walk
from .node2vec import node2vec

def embedding_factory(args):
    if args.algorithm == "deep_walk":
        deep_walk();
    elif args.algorithm == "node2vec":
        node2vec();
    else:
        raise Exception("algorithm not supported");
