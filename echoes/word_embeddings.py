import argparse
import configparser
import glob
import os
import shutil
import json

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models import FastText, Word2Vec
from gensim.similarities.index import AnnoyIndexer


class Sentences:
    def __init__(self, input_file):
        self.input_file = input_file
    
    def __iter__(self):
        with open(self.input_file) as f:
            for line in f:
                for sentence in json.loads(line)['sentences']:
                    yield [t.lower() for t in sentence['tokens']]


def main():
    parser = argparse.ArgumentParser(description='Trains word embeddings')
    parser.add_argument('--config_file', type=str,
                        default='configs/echoes_local.config',
                        help='location of the configuration file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)

    print(config['word']['model_dir'])

    sentences = Sentences(input_file=config['general']['corpus_file'])
    try:
        shutil.rmtree(config['word']['model_dir'])
    except FileNotFoundError:
        pass
    os.mkdir(config['word']['model_dir'])

    logging.info('Building fasttext model...')
    model = FastText(sentences,
                     size=int(config['word']['size']),
                     window=int(config['word']['window']),
                     min_count=int(config['word']['min_count']),
                     iter=int(config['word']['epochs']),
                     workers=int(config['word']['workers']))
    model.init_sims()
    model.save(f"{config['word']['model_dir']}/ft_model")
    logging.info(f"Saved fasttext model under {config['word']['model_dir']}")

    logging.info('Building word2vec model...')
    model = Word2Vec(sentences,
                     size=int(config['word']['size']),
                     window=int(config['word']['window']),
                     min_count=int(config['word']['min_count']),
                     iter=int(config['word']['epochs']),
                     workers=int(config['word']['workers']))
    model.init_sims()
    annoy_index = AnnoyIndexer(model, 100)
    annoy_index.save(f"{config['word']['model_dir']}/annoy_model")
    model.save(f"{config['word']['model_dir']}/w2v_model")
    logging.info(f"Saved word2vec model under {config['word']['model_dir']}")

if __name__ == '__main__':
    main()
