import os

from elasticsearch import Elasticsearch
import flask
import flask_sqlalchemy

from .neighbors import WordNeighbors
from .neighbors import PhraseNeighbors

import config


app = flask.Flask(__name__)
app.config.from_object(config.Config)

db = flask_sqlalchemy.SQLAlchemy(app)

app.elasticsearch = Elasticsearch([app.config['ELASTICSEARCH_URL']])
app.semantic_neighbors = WordNeighbors(app.config['MODEL_DIR'])

if os.path.exists(os.path.join(app.config['DATA_DIR'], 'faiss_db')):
    app.sentence_neighbors = PhraseNeighbors(app.config['DATA_DIR'])

from . import views, models, search
