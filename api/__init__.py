import configparser
import os.path

from gensim.models import FastText, Word2Vec
from gensim.similarities.index import AnnoyIndexer

import flask
from flask import request, jsonify

import whoosh.index
from whoosh import qparser
from whoosh.highlight import UppercaseFormatter


config = configparser.ConfigParser()
config.read('echoes.config')

ft_model = FastText.load(os.path.join(config.get('word', 'model_dir'), 'ft_model'))
w2v_model = Word2Vec.load(os.path.join(config.get('word', 'model_dir'), 'w2v_model'))

annoy_index = AnnoyIndexer()
annoy_index.load(os.path.join(config.get('word', 'model_dir'), 'annoy_model'))
annoy_index.model = w2v_model

concordance_dir = config.get('concordance', 'indexdir')
concordancer = whoosh.index.open_dir(concordance_dir)

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
        e = 'Error: No w-field provided. Please specify a non-empty word.'
        return jsonify({'status': 'fail', 'message': str(e), 'code': 500})
        
    topn = int(request.args.get('topn', topn))

    if w in w2v_model:
        vector = w2v_model[w]
        neighbors = w2v_model.most_similar([vector], topn=topn, indexer=annoy_index)
    else:
        try:
            neighbors = ft_model.most_similar(w, topn=topn)
        except KeyError:
            # no relevant ngrams at all:
            neighbors = []
    if neighbors:
        neighbors = [{'word': w, 'sim': d} for w, d in neighbors]
    return jsonify({'status': 'OK', 'results': neighbors})

@app.route('/api/concordance', methods=['GET'])
def concordance(limit=20):
    if 'w' in request.args and request.args['w'].strip():
        w = request.args['w'].strip()
    else:
        e = 'Error: No w-field provided. Please specify a non-empty word.'
        return jsonify({'status': 'fail', 'message': str(e), 'code': 500})

    topn = int(request.args.get('topn', limit))

    json_out = []
    qp = qparser.QueryParser("text", schema=concordancer.schema)
    with concordancer.searcher() as searcher:
        results = searcher.search(qp.parse(w), limit=topn)
        results.formatter = UppercaseFormatter()
        for hit in results:
            with open(hit["path"]) as f:
                contents = f.read()
            json_out.append({
                'path': hit['path'],
                'snippet': hit.highlights("text", text=contents)
            })
    return jsonify(json_out)
