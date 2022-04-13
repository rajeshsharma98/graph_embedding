from .embedding_algorithm import Embedding
from walks.factory import walker
from .word2Vec import word2VecEmbedding

class DeepWalk(Embedding):

    def __init__(self, args):
        self.embedding = None;
        self.walk_len = args.walk_len
        self.walks = args.walks
        self.walker = args.walker
        self.set_embedding()

    def set_embedding(self):
        # perform walks
        
        ## TODO: remove this
        adj_list = [["0", "1", "2"],["5", "6"], ["5", "1"], ["1", "2"], ["3", "4"], ["6", "1"], ["5", "2"]]

        walks = walker(adj_list, self.walks, self.walk_len, self.walker)

        # print(walks)
        
        # word2vec on generated walks
        self.embedding = word2VecEmbedding(walks);

        print(self.embedding.wv['0']);

    def get_embedding(self):
        return self.embedding;


def deep_walk(args):
    print("running deep walk...")
    model = DeepWalk(args);
    return model.get_embedding();
