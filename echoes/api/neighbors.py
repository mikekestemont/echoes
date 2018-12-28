import os
import json

import numpy as np

from gensim.models import FastText, Word2Vec
from gensim.similarities.index import AnnoyIndexer

from elmoformanylangs import Embedder
import faiss

from syntok.tokenizer import Tokenizer
tokenizer = Tokenizer()


class WordNeighbors:
    def __init__(self, model_dir):
        self.ft_model = FastText.load(os.path.join(model_dir, 'ft_model'))
        self.w2v_model = Word2Vec.load(os.path.join(model_dir, 'w2v_model'))
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


class PhraseNeighbors:
    def __init__(self, model_dir):
        self.faiss_db = faiss.read_index(os.path.join(model_dir, 'faiss_db'))
        with open(os.path.join(model_dir, 'faiss_lookup.json')) as f:
            self.faiss_lookup = json.loads(f.read())
        self.elmo = Embedder(os.path.join(model_dir, 'elmo_nl'))
        
    def query(self, s, topn):
        s = [tokenizer.tokenize(s)]
        X = self.elmo.sents2elmo(s)
        X = np.array([x.mean(axis=0) for x in X])    

        distances, indices = self.faiss_db.search(X, k=topn)

        return [(self.faiss_lookup[i], d) for i, d in zip(indices, distances)]
