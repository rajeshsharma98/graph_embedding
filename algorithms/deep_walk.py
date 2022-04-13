from .embedding_algorithm import Embedding
from walks.factory import walker
from .word2Vec import word2VecEmbedding
from utils import adj_list_reader

class DeepWalk(Embedding):

    def __init__(self, args):
        self.embedding = None;
        self.walk_len = args.walk_len
        self.walks = args.walks
        self.walker = args.walker
        self.adj_list = adj_list_reader(args.adj_list)
        self.set_embedding()

    def set_embedding(self):
        # perform walks
        walks = walker(self.adj_list, self.walks, self.walk_len, self.walker)
        
        # word2vec on generated walks
        self.embedding = word2VecEmbedding(walks);

    def get_embedding(self):
        return self.embedding;


def deep_walk(args):
    print("running deep walk...")
    model = DeepWalk(args);
    return model.get_embedding();
