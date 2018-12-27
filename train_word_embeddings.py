import argparse
import configparser
import glob
import os
import shutil

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from nltk import word_tokenize, sent_tokenize
from gensim.models import FastText, Word2Vec
from gensim.similarities.index import AnnoyIndexer


class Sentences:
    def __init__(self, input_dir, preprocess=True, max_files=None):
        self.input_dir = input_dir
        self.tmp_dir = input_dir + '-tmp'

        if preprocess:
            try:
                shutil.rmtree(self.tmp_dir)
            except FileNotFoundError:
                pass
            os.mkdir(self.tmp_dir)

            fns = glob.glob(f'{self.input_dir}/*.txt')
            if max_files:
                fns = list(fns)[:max_files]
            for fn in fns:
                print(fn)
                newp = f'{self.tmp_dir}/{os.path.basename(fn)}'
                with open(fn) as f, open(newp, 'w') as newf:
                    for sentence in sent_tokenize(f.read()):
                        newf.write(' '.join(word_tokenize(sentence)) + '\n')
    
    def __iter__(self):
        for fn in list(glob.glob(f'{self.tmp_dir}/*.txt')):
            with open(fn) as f:
                for line in f:
                    yield line.strip().lower().split()


def main():
    parser = argparse.ArgumentParser(description='Trains word embeddings')
    parser.add_argument('--config_file', type=str,
                        default='echoes.config',
                        help='location of the configuration file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)

    sentences = Sentences(input_dir=config['general']['input_dir'],
                          max_files=int(config['general']['max_files']),
                          preprocess=bool(config['general']['preprocess']))
    try:
        shutil.rmtree(config['word']['model_dir'])
    except FileNotFoundError:
        pass
    os.mkdir(config['word']['model_dir'])

    model = FastText(sentences,
                     size=int(config['word']['size']),
                     window=int(config['word']['window']),
                     min_count=int(config['word']['min_count']),
                     iter=int(config['word']['epochs']),
                     workers=int(config['word']['workers']))
    model.init_sims()
    model.save(f"{config['word']['model_dir']}/ft_model")

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

if __name__ == '__main__':
    main()