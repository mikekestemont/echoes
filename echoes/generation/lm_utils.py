import json
import collections
import json
import torch
import numpy as np

SYMBOLS = PAD, LB, UNK = '<pad>', '<l>', '<unk>'


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    return data.view(bsz, -1).t().contiguous()

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def lines_from_jsonl(path):
    with open(path) as f:
        for line in f:
            for sentence in json.loads(line)['sentences']:
                yield sentence['sentence']


class Vocabulary:
    def __init__(self, min_cnt = 0, char2idx = None,
                 idx2char =  None):
        self.min_cnt = min_cnt
        self.char2idx = char2idx if char2idx is not None else {}
        if idx2char is not None:
            self.idx2char = {int(k):v for k, v in idx2char.items()}
        else:
            self.idx2char = {}
        self.fitted = len(self.char2idx) > 0

    def fit(self, lines):
        counter = collections.Counter()
        for line in lines:
            counter.update(line)

        self.char2idx = {}
        for symb in list(SYMBOLS) + sorted(k for k, v in counter.most_common()
                                     if v >= self.min_cnt):
            self.char2idx[symb] = len(self.char2idx)

        self.idx2char = {i: s for s, i in self.char2idx.items()}
        return self
    
    def transform(self, chars):
        return [self.char2idx.get(c, self.char2idx['<unk>']) for c in chars]

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump(
                {
                    'min_cnt': self.min_cnt,
                    'idx2char': self.idx2char,
                    'char2idx': self.char2idx
                },
                f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            params = json.load(f)
        return cls(**params)

class CorpusReader:
    def __init__(self, path, conds, vocab):
        self.path = path
        self.conds = conds
        self.vocab = vocab
    
    def get_batches(self, batch_size, bptt):
        for line in open(self.path):
            if not line.strip():
                continue

            # set conds here!
            
            novel = json.loads(line)
            chars = []
            for sentence in novel['sentences']:
                chars += list(sentence['sentence'].strip()) + ['<lb>']
            
            chars = torch.LongTensor(self.vocab.transform(chars))
            chars = batchify(chars, batch_size)

            for batch_idx, i in enumerate(range(0, chars.size(0) - 1, bptt)):
                source, target = get_batch(chars, i, bptt)
                yield source, target
    
