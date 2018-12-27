from flask import request, jsonify

from .search import query_index
from .models import Text

from api import app


@app.route('/api/word', methods=['GET'])
def neighbors(limit=10):
    """
    locally:
    http://127.0.0.1:5000/api/word?w=kat&limit=4

    server:
    http://mikekestemont.pythonanywhere.com/api/word?w=kat&limit=4
    """

    if 'w' in request.args and request.args['w'].strip():
        w = request.args['w'].strip()
    else:
        e = 'Error: No w-field provided. Please specify a non-empty word.'
        return jsonify({'status': 'fail', 'message': e, 'code': 500})
        
    limit = int(request.args.get('limit', limit))
    neighbors = app.semantic_neighbors.query(w, limit)
    if neighbors:
        neighbors = [{'word': w, 'sim': d} for w, d in neighbors]
    return jsonify({'status': 'OK', 'results': neighbors})


@app.route('/api/concordance', methods=['GET'])
def concordance(limit=5):
    if 'w' in request.args and request.args['w'].strip():
        w = request.args['w'].strip()
    else:
        e = 'Error: No w-field provided. Please specify a non-empty word.'
        return jsonify({'status': 'fail', 'message': e, 'code': 500})

    limit = int(request.args.get('limit', limit))

    hits, snippets, total = query_index('echoes-texts', w, limit=limit)
    return jsonify({'hits': hits, 'snippets': snippets, 'total': total})


@app.route('/api/sentence', methods=['GET'])
def sentence():
    document_id = int(request.args['did'])
    sentence_id = int(request.args['sid'])
    sentence = Text.query.get(document_id)
    return jsonify({'sentence': sentence.get(sentence_id)})
