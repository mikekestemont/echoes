import argparse

import torch

from lm_utils import Vocabulary
from modelling import LM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model_zoo')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='initial learning rate')
    parser.add_argument('--seed', type=str, default='',
                        help='initial learning rate')
    parser.add_argument('--num', type=int, default=10,
                        help='initial learning rate')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    args = parser.parse_args()
    print(args)
    
    device = torch.device('cuda' if args.cuda else 'cpu')

    vocab = Vocabulary.load(f'{args.model_path}/vocab.json')
    lm = torch.load(f'{args.model_path}/lm.pt').to(device)

    in_ = torch.randint(len(vocab.idx2char), (1, 1), dtype=torch.long).to(device)
    hid_ = None

    if args.seed:
        ints = torch.tensor(vocab.transform(args.seed)).to(device)
        for char_idx in range(len(ints) - 1):
            in_.fill_(ints[char_idx])
            _, hid_ = lm.forward(in_, hid_)
        in_.fill_(ints[-1])
    
    
    hid_ = hid_.repeat((1, args.num, 1))
    in_ = in_.repeat((1, args.num))
    
    hypotheses = [[] for i in range(args.num)]

    for i in range(100):
        output, hid_ = lm.forward(in_, hid_)
        char_weights = output.squeeze().div(args.temperature).exp().cpu()
        char_idx = torch.multinomial(char_weights, 1)
        in_ = torch.t(char_idx)
        for idx, char in enumerate(char_idx.squeeze()):
            hypotheses[idx].append(vocab.idx2char[char.item()])

    for hyp in hypotheses:
        print(''.join(hyp))


if __name__ == '__main__':
    main()