import os

from elasticsearch import Elasticsearch
import flask
import flask_sqlalchemy

from .neighbors import SemanticNeighbors

import config


app = flask.Flask(__name__)
app.config.from_object(config.Config)

db = flask_sqlalchemy.SQLAlchemy(app)

app.elasticsearch = Elasticsearch([app.config['ELASTICSEARCH_URL']])
app.semantic_neighbors = SemanticNeighbors(app.config['MODEL_DIR'])

from api import views, models, search
