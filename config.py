import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # data and models paths
    MODEL_DIR = os.path.join(basedir, 'data/word_models')
    CORPUS_DIR = os.path.join(basedir, 'data/example_corpus')

    # Search and database URLs
    ELASTICSEARCH_URL = 'http://localhost:9200'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'echoes.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Elasticsearch settings
    NUMBER_OF_FRAGMENTS = 5
    FRAGMENT_SIZE = 100
