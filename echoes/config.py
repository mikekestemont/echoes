import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # data and models paths
    DATA_DIR = os.path.join(basedir, '../data')
    MODEL_DIR = os.path.join(basedir, '../data/word_models')
    CORPUS_DIR = os.path.join(basedir, '../data/corpus')

    CORPUS_FILE = os.path.join(basedir, '../data/corpus.jsonl')

    # Search and database URLs
    ELASTICSEARCH_URL = 'http://localhost:9200'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, '../data/echoes.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Elasticsearch settings
    NUMBER_OF_FRAGMENTS = 5
    FRAGMENT_SIZE = 100

    # Swagger
    SWAGGER = {'title': 'Echoes API'}
