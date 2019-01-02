import json

SYMBOLS = PAD, LB, UNK = '<pad>', '<l>', '<unk>'

def lines_from_jsonl(path):
    with open(path) as f:
        for line in f:
            for sentence in json.loads(line)['sentences']:
                yield sentence['sentence']

import collections
import json

import numpy as np

SYMBOLS = PAD, BOS, EOS, UNK = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

class Vocabulary:
    def __init__(self, min_cnt = 0, char2idx = None,
                 idx2char =  None):
        self.min_cnt = min_cnt
        self.char2idx = char2idx if char2idx is not None else {}
        self.idx2char = idx2char if idx2char is not None else {}
        self.fitted = len(self.char2idx) > 0

    def fit(self, lines):
        counter = collections.Counter()
        for line in lines:
            counter.update(line)

        self.char2idx = {}
        for symb in SYMBOLS + sorted(k for k, v in counter.most_common()
                                     if v >= self.min_cnt):
            self.char2idx[symb] = len(self.char2idx)

        self.idx2char = {i: s for s, i in self.char2idx.items()}
        return self

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump(
                {
                    'min_cnt': self.min_cnt,
                    'idx2char': self.char2idx,
                    'char2idx': self.char2idx
                },
                f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            params = json.load(f)
        return cls(**params)



                