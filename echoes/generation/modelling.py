import os
import json
import shutil
import time
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class LM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, vocab, layers, emb_dim, bptt, modelname,
                 hidden_dim, cond_dim, tie_weights=False, dropout=0.5):
        super(LM, self).__init__()
        self.vocab_size = len(vocab.char2idx)
        self.modelname = modelname
        self.drop = nn.Dropout(dropout)
        self.layers = layers
        self.emb_dim = emb_dim
        self.bptt = bptt
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.tie_weights = tie_weights
        
        self.encoder = nn.Embedding(self.vocab_size, self.emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim, layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
    
    def train_model(self, train, dev, vocab, epochs, batch_size, bptt,
                    lr, device, clip, log_interval, temperature):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        def epoch():
            self.train()

            total_loss = 0.
            start_time = time.time()
            hidden = None

            for batch, (source, target) in enumerate(train.get_batches(batch_size, bptt)):
                source, target = source.to(device), target.to(device)

                self.zero_grad()
                output, hidden = self.forward(source, hidden)
                loss = criterion(output.view(-1, self.vocab_size), target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()

                total_loss += loss.item()
                hidden = repackage_hidden(hidden)

                if batch % log_interval == 0 and batch > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                        'loss {:5.6f} | ppl {:8.2f}'.format(
                        epoch_idx, batch * batch_size, lr,
                        elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()

                    with torch.no_grad():
                        hid_ = None
                        in_ = torch.randint(self.vocab_size, (1, 1), dtype=torch.long).to(device)

                        for i in range(100):
                            output, hid_ = self.forward(in_, hid_)
                            char_weights = output.squeeze().div(temperature).exp().cpu()
                            char_idx = torch.multinomial(char_weights, 1)[0]
                            in_.fill_(char_idx)
                            char = vocab.idx2char[char_idx.item()]
                            print(char, end='')

                    print('\n')
        
        def evaluate():
            self.eval()
            losses = []
            hidden = None

            with torch.no_grad():
                
                for batch, (source, target) in enumerate(dev.get_batches(batch_size, bptt)):
                    source, target = source.to(device), target.to(device)
                    output, hidden = self.forward(source, hidden)
                    output_flat = output.view(-1, self.vocab_size)
                    losses.append(len(source) * criterion(output_flat, target).item())
                    hidden = repackage_hidden(hidden)

            return sum(losses) / len(losses)

        best_val_loss = None
        
        try:
            for epoch_idx in range(1, epochs+1):
                epoch_start_time = time.time()
                epoch()
                val_loss = evaluate()

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.6f}'.format(epoch_idx, (time.time() - epoch_start_time),
                                                val_loss, math.exp(val_loss)))
                print('-' * 89)

                if not best_val_loss or val_loss < best_val_loss:
                    with open(self.modelname + '.pt', 'wb') as f:
                        torch.save(self, f)
                    print('>>> saving model')    
                    best_val_loss = val_loss
                elif val_loss >= best_val_loss:
                    lr *= 0.5
                    print(f'>>> lowering learning rate to {lr}')
                    for g in optimizer.param_groups:
                        g['lr'] = lr
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')