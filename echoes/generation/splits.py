import argparse
import shutil
import os
from collections import Counter

from sklearn.model_selection import train_test_split as split

def main():
    parser = argparse.ArgumentParser(description='Splits available data in train vs dev')
    parser.add_argument('--input_file', type=str,
                        default='/Users/mikekestemont/GitRepos/echoes/data/corpus.jsonl',
                        help='location of the full data file')
    parser.add_argument('--split_dir', type=str,
                        default='/Users/mikekestemont/GitRepos/echoes/data/lm_splits',
                        help='location of the train-dev files')
    parser.add_argument('--train_prop', type=float,
                        default=.8,
                        help='Proportion of training items (dev and test are equal-size)')
    parser.add_argument('--seed', type=int,
                        default=43438,
                        help='Random seed')
    args = parser.parse_args()
    print(args)

    try:
        shutil.rmtree(args.split_dir)
    except FileNotFoundError:
        pass
    os.mkdir(args.split_dir)

    ntotal = 0
    with open(args.input_file) as f:
        for line in f:
            if not line.strip():
                continue
            ntotal += 1
    
    print(f'Total number of files: {ntotal}')

    train_idxs, _ = split(range(ntotal),
                       train_size=args.train_prop,
                       shuffle=True,
                       random_state=args.seed)
    train_idxs = set(train_idxs)
    
    with open(args.input_file, 'r') as origf, \
         open(f'{args.split_dir}/train.jsonl', 'w') as trainf, \
         open(f'{args.split_dir}/dev.jsonl', 'w') as devf:
        for idx, line in enumerate(origf):
            if idx in train_idxs:
                trainf.write(line)
            else:
                devf.write(line)    

if __name__ == '__main__':
    main()