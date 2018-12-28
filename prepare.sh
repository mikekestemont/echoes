#!/bin/bash

# remove previous DB
rm data/echoes.db
curl -XDELETE 'http://localhost:9200/echoes-texts/'

# run preprocessing engines
python echoes/preprocess_corpus.py
python echoes/word_embeddings.py
python echoes/build_index.py
