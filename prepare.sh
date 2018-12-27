#!/bin/bash

# remove previous DB
rm data/echoes.db
curl -XDELETE 'http://localhost:9200/echoes-texts/'

# run preprocessing engines
python src/preprocess_corpus.py
python src/word_embeddings.py
python src/build_index.py
