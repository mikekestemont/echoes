import argparse
import configparser
import glob
import json
import os
import re
import math

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import pandas as pd
import syntok.segmenter as segmenter
from tqdm import tqdm

WHITESPACE = re.compile(r'\s+')

def main():
    parser = argparse.ArgumentParser(description='Preprocess corpus')
    parser.add_argument('--config_file', type=str,
                        default='configs/echoes_local.config',
                        help='location of the configuration file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)

    orig_dir = config['general']['orig_dir']
    logging.info(f'Preprocessing corpus files under: {orig_dir}')
    corpus_file = config['general']['corpus_file']
    max_files = int(config['general']['max_files'])

    fns = glob.glob(f'{orig_dir}/*.txt')
    if max_files > -1:
        fns = list(fns)[:max_files]
    
    metadata = pd.read_excel(config['general']['metadata'])
    metadata.fillna('', inplace=True)
    metadata.set_index('filepath', inplace=True)
    
    with open(corpus_file, 'w') as corpusf:
        for fn in tqdm(fns):
            with open(fn, encoding='utf-8-sig') as f:
                # try to extract metadata:
                md = {'author': 'Unknown', 'title': 'Unknown'}
                try:
                    r = metadata.loc[os.path.basename(fn).replace('.txt', '')]
                    if r['author:lastname']:
                        md['author'] = r['author:lastname']
                    if r['title:detail']:
                        md['title'] = r['title:detail']
                except KeyError:
                    pass
                
                # extract sentences:
                sentences = []
                for paragraph in segmenter.process(f.read()): 
                    for sentence in paragraph:
                        # get original sentence:
                        string_repr = ''
                        for token in sentence:
                            string_repr += f'{token.spacing}{token.value}'
                        string_repr = WHITESPACE.sub(' ', string_repr).strip()

                        # get individual tokens:
                        tokens = [t.value.strip() for t in sentence]
                        tokens = [t for t in tokens if t]

                        if tokens and string_repr:
                            sentences.append({'sentence': string_repr,
                                              'tokens': tokens})
                if sentences:
                    sentences = {'metadata': md,
                                 'sentences': sentences}
                    corpusf.write(json.dumps(sentences) + '\n')

    logging.info(f'Finished tokenizing corpus to: {corpus_file}')

if __name__ == '__main__':
    main()
