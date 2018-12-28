import os
import argparse
import json
import logging

from elmoformanylangs import Embedder
import faiss
import tqdm
import numpy as np

from api import app, db, models, search


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

db.create_all()

elmo = Embedder(os.path.join(app.config['DATA_DIR'], 'elmo_nl'))
faiss_db = faiss.IndexFlatL2(1024)
faiss_lookup = []

logging.info(f"Adding files to database from {app.config['CORPUS_FILE']}")

with open(app.config['CORPUS_FILE']) as f:
    for text_id, line in enumerate(tqdm.tqdm(f, desc='Adding documents to database')):
        json_text = json.loads(line)
        author = json_text['metadata']['author']
        title = json_text['metadata']['title']

        # add sentence string to database:
        text = models.Text(source=title, author=author,
                           text=[s['sentence'] for s in json_text['sentences']][:100])
        db.session.add(text)

        # add sentence embeddings to faiss:
        tokenized = [s['tokens'] for s in json_text['sentences']][:100]
        X = elmo.sents2elmo(tokenized)
        X = np.array([x.mean(axis=0) for x in X])
        faiss_db.add(X)

        for i in range(len(X)):
            faiss_lookup.append((text_id, i))

        assert faiss_db.ntotal == len(faiss_lookup)

db.session.commit()
faiss.write_index(faiss_db, os.path.join(app.config['DATA_DIR'], 'faiss_db'))

with open(os.path.join(app.config['DATA_DIR'], 'faiss_lookup.json'), 'w') as f:
    f.write(json.dumps(faiss_lookup, indent=4))

logging.info("Adding files to elasticsearch")

for text in tqdm.tqdm(models.Text.query.all(), desc='Indexing with Elasticsearch'):
    search.index_document('echoes-texts', text)
logging.info('Done indexing files')
