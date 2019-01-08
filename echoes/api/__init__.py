import os

from elasticsearch import Elasticsearch
import flask
import flask_sqlalchemy
from flasgger import Swagger

from .neighbors import WordNeighbors
from .neighbors import PhraseNeighbors
from .neighbors import Completer

import config


app = flask.Flask(__name__)
app.config.from_object(config.Config)
swagger = Swagger(app)

db = flask_sqlalchemy.SQLAlchemy(app)

app.elasticsearch = Elasticsearch([app.config['ELASTICSEARCH_URL']])
app.semantic_neighbors = WordNeighbors(app.config['MODEL_DIR'])

if os.path.exists(os.path.join(app.config['DATA_DIR'], 'faiss_db')):
    app.sentence_neighbors = PhraseNeighbors(app.config['DATA_DIR'])

if os.path.exists(app.config['LM_DIR']):
    app.completer = Completer(app.config['LM_DIR'])

from . import views, models, search
