import os

from gensim.models import FastText, Word2Vec
from gensim.similarities.index import AnnoyIndexer


class SemanticNeighbors:
    def __init__(self, model_dir):
        if os.path.exists(os.path.join(model_dir, 'ft_model')):
            self.ft_model = FastText.load(os.path.join(model_dir, 'ft_model'))
        if os.path.exists(os.path.join(model_dir, 'w2v_model')):
            self.w2v_model = Word2Vec.load(os.path.join(model_dir, 'w2v_model'))
        if os.path.exists(os.path.join(model_dir, 'annoy_model')):
            self.annoy_index = AnnoyIndexer()
            self.annoy_index.load(os.path.join(model_dir, 'annoy_model'))

    def query(self, w, topn):
        if w in self.w2v_model:
            vector = self.w2v_model[w]
            neighbors = self.w2v_model.most_similar(
                [vector], topn=topn, indexer=self.annoy_index)
        else:
            try:
                neighbors = self.ft_model.most_similar(w, topn=topn)
            except KeyError:
                neighbors = []
        return neighbors
