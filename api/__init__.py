import configparser
import os.path

from gensim.models import FastText, Word2Vec
from gensim.similarities.index import AnnoyIndexer

import flask
from flask import request, jsonify


config = configparser.ConfigParser()
config.read('echoes_server.config')

ft_model = FastText.load(os.path.join(config.get('word', 'model_dir'), 'ft_model'))
w2v_model = Word2Vec.load(os.path.join(config.get('word', 'model_dir'), 'w2v_model'))

annoy_index = AnnoyIndexer()
annoy_index.load(os.path.join(config.get('word', 'model_dir'), 'annoy_model'))
annoy_index.model = w2v_model

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return '''<h1>API Home</h1>'''

@app.route('/api/word', methods=['GET'])
def neighbors(topn=10):
    """
    locally:
    http://127.0.0.1:5000/api/word?w=kat&topn=4

    server:
    http://mikekestemont.pythonanywhere.com/api/word?w=kat&topn=4
    """

    if 'w' in request.args and request.args['w'].strip():
        w = request.args['w'].strip()
    else:
        return 'Error: No w-field provided. Please specify a non-empty word.'
        
    topn = int(request.args.get('topn', topn))

    if w in w2v_model:
        vector = w2v_model[w]
        neighbors = w2v_model.most_similar([vector], topn=topn, indexer=annoy_index)
    else:
        try:
            neighbors = ft_model.most_similar(w, topn=topn)
        except KeyError:
            # no relevant ngrams at all:
            neighbors = None
    
    neighbors = [{'word': w, 'sim': d} for w, d in neighbors]
    return jsonify(neighbors)