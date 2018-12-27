import os

import syntok.segmenter as segmenter
import tqdm

from api import app, db, models, search


def iter_sentences(text):
    for paragraph in segmenter.process(text): 
        for sentence in paragraph:
            string_repr = ''
            for token in sentence:
                string_repr += f'{token.spacing}{token.value}'
            if string_repr.strip():
                yield string_repr.strip()


db.create_all()


n_files = len(os.listdir(app.config['CORPUS_DIR']))
for entry in tqdm.tqdm(os.scandir(app.config['CORPUS_DIR']), total=n_files,
                       desc='Adding documents to database'):
    if entry.path.endswith('.txt'):
        with open(entry.path) as f:
            # TODO: add source and author
            text = models.Text(source="Unknown", author="Anonymous",
                               text=list(iter_sentences(f.read())))
            db.session.add(text)
db.session.commit()

for text in tqdm.tqdm(models.Text.query.all(), desc='Indexing with Elasticsearch'):
    search.index_document('echoes-texts', text)
