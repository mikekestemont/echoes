import configparser
import os

import tqdm
import whoosh.index
import whoosh.fields
from whoosh.fields import TEXT, ID
from whoosh.analysis import SimpleAnalyzer


schema = whoosh.fields.Schema(
    path=ID(stored=True),
    text=TEXT(analyzer=SimpleAnalyzer(), phrase=True)
)

config = configparser.ConfigParser()
config.read('echoes.config')
index_dir = config.get('concordance', 'indexdir')
input_dir = config.get('general', 'input_dir')

if not os.path.exists(index_dir):
    os.mkdir(index_dir)
    whoosh.index.create_in(index_dir, schema)

ix = whoosh.index.open_dir(index_dir)
writer = ix.writer(limitmb=1024)
n_files = len(os.listdir(input_dir))
for entry in tqdm.tqdm(os.scandir(input_dir), total=n_files):
    if entry.path.endswith('.txt'):
        with open(entry.path) as f:
            writer.add_document(path=entry.path, text=f.read())
writer.commit()
