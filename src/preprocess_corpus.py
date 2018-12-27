import argparse
import configparser
import glob
import os
import shutil

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import syntok.segmenter as segmenter
from tqdm import tqdm


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
    input_dir = config['general']['input_dir']
    max_files = int(config['general']['max_files'])
    
    try:
        shutil.rmtree(input_dir)
    except FileNotFoundError:
        pass
    os.mkdir(input_dir)

    fns = glob.glob(f'{orig_dir}/*.txt')
    if max_files:
        fns = list(fns)[:max_files]
    
    for fn in tqdm(fns):
        newp = f'{input_dir}/{os.path.basename(fn)}'
        with open(fn) as f, open(newp, 'w') as newf:
            for paragraph in segmenter.process(f.read()): 
                for sentence in paragraph:
                    newf.write(' '.join([t.value for t in sentence]) + '\n')

    logging.info(f'Finished tokenizing corpus to: {input_dir}')

if __name__ == '__main__':
    main()