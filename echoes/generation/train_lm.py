import argparse
import shutil
import os

import torch

from lm_utils import lines_from_jsonl, Vocabulary, CorpusReader
from modelling import LM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--split_dir', type=str, default='/Users/mikekestemont/GitRepos/echoes/data/lm_splits')
    parser.add_argument('--conds')
    parser.add_argument('--bptt', type=int, default=30)
    parser.add_argument('--min_cnt', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='model_zoo')

    # model
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--cond_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=250)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.2)

    # train
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='initial learning rate')
    
    args = parser.parse_args()
    print(args)

    try:
        shutil.rmtree(args.model_path)
    except FileNotFoundError:
        pass
    os.mkdir(args.model_path)

    device = torch.device('cuda' if args.cuda else 'cpu')

    vocab = Vocabulary(min_cnt=args.min_cnt)
    vocab.fit(lines_from_jsonl(f'{args.split_dir}/train.jsonl'))
    vocab.dump(f'{args.model_path}/vocab.json')
    
    conds = None
    if args.conds:
        conds = set(args.conds.split(','))
    
    train = CorpusReader(path=f'{args.split_dir}/train.jsonl', vocab=vocab, conds=conds)
    dev = CorpusReader(path=f'{args.split_dir}/dev.jsonl', vocab=vocab, conds=conds)

    #for batch in train.get_batches(args.batch_size, args.bptt):
    #    print(batch)


    print("Building model")
    lm = LM(vocab=vocab, layers=args.layers, emb_dim=args.emb_dim,
            bptt=args.bptt, hidden_dim=args.hidden_dim,
            model_dir=args.model_path,
            cond_dim=args.cond_dim, tie_weights=args.tie_weights,
            dropout=args.dropout, modelname='XXX')
    print(lm)
    print("Model parameters: {}".format(sum(p.nelement() for p in lm.parameters())))
    lm.train_model(train, dev, vocab, lr=args.lr, clip=args.clip, temperature=args.temperature,
                   device=device, log_interval=args.log_interval,
                   epochs=args.epochs, batch_size=args.batch_size, bptt=args.bptt)