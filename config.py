import os

basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)

class Config:
    MODEL_DIR = os.path.join(basedir, 'data/word_models')
    CORPUS_DIR = os.path.join(basedir, 'data/example_corpus')
    ELASTICSEARCH_URL = 'http://localhost:9200'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'echoes.db')
