import os
import json

import numpy as np

from gensim.models import FastText, Word2Vec
from gensim.similarities.index import AnnoyIndexer

from elmoformanylangs import Embedder
import faiss

import torch
from .. generation.lm_utils import vocabulary

from syntok.tokenizer import Tokenizer


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
        self.tokenizer = Tokenizer()

    def query(self, s, topn):
        s = [w.value for w in self.tokenizer.tokenize(s)]
        X = self.elmo.sents2elmo([s])
        X = np.array([x.mean(axis=0) for x in X])

        distances, indices = self.faiss_db.search(X, k=topn)

        return [(self.faiss_lookup[i], d) for i, d in zip(indices[0], distances[0])]

class Completer:
    def __init__(self, model_dir, cuda=False):
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.vocab = Vocabulary.load(f'{model_dir}/vocab.json')
        self.lm = torch.load(f'{model_dir}/lm.pt').to(self.device)

    def query(self, s, num_alternatives=5, sugg_len=60, temp=.35):
        in_ = torch.randint(len(self.vocab.idx2char), (1, 1), dtype=torch.long)
        in_ = in_.to(self.device)
        hid_ = None

        if s:
            ints = torch.tensor(self.vocab.transform(s)).to(self.device)
            for char_idx in range(len(ints) - 1):
                in_.fill_(ints[char_idx])
                _, hid_ = self.lm.forward(in_, hid_)
            in_.fill_(ints[-1])
            hid_ = hid_.repeat((1, num_alternatives, 1))
        
        in_ = in_.repeat((1, num_alternatives))
        
        hypotheses = [[] for i in range(num_alternatives)]

        for _ in range(sugg_len):
            output, hid_ = self.lm.forward(in_, hid_)
            char_weights = output.squeeze().div(temp).exp().cpu()
            char_idx = torch.multinomial(char_weights, 1)
            in_ = torch.t(char_idx)
            for idx, char in enumerate(char_idx.squeeze()):
                hypotheses[idx].append(self.vocab.idx2char[char.item()])

        return [''.join(h) for h in hypotheses]
