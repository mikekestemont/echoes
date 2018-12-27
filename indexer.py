import configparser
import os
import tqdm

from api import app, db, models, search

db.create_all()

n_files = len(os.listdir(app.config['CORPUS_DIR']))
for entry in tqdm.tqdm(os.scandir(app.config['CORPUS_DIR']), total=n_files,
                       desc='Adding documents to database'):
    if entry.path.endswith('.txt'):
        with open(entry.path) as f:
            text = models.Text(source="Unknown", author="Anonymous", text=f.read())
            db.session.add(text)
db.session.commit()

for text in tqdm.tqdm(models.Text.query.all(), desc='Indexing with Elasticsearch'):
    search.index_document('echoes-texts', text)
