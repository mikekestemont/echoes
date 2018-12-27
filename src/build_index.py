import os
import json
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import tqdm

from api import app, db, models, search

def main():
    db.create_all()

    logging.info(f"Adding files to database from {app.config['CORPUS_FILE']}")

    with open(app.config['CORPUS_FILE']) as f:
        for line in tqdm.tqdm(f, desc='Adding documents to database'):
            text = json.loads(line)
            author = text['metadata']['author']
            title = text['metadata']['title']
            
            text = models.Text(source=title, author=author,
                               text=[s['sentence'] for s in text['sentences']])
            db.session.add(text)
    
    db.session.commit()

    logging.info(f"Adding files to elasticsearch")
    for text in tqdm.tqdm(models.Text.query.all(), desc='Indexing with Elasticsearch'):
        search.index_document('echoes-texts', text)
    logging.info(f'Done indexing files')


if __name__ == '__main__':
    main()