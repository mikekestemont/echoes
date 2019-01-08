from flask import request, jsonify
from flasgger import swag_from

from .search import query_index
from .models import Text

from . import app


@app.route('/api/word', methods=['GET'])
@swag_from('../openapi/word.yml')
def word_neighbors():
    if 'q' in request.args and request.args['q'].strip():
        query = request.args['q'].strip()
    else:
        e = 'Error: No q-field provided. Please specify a non-empty word.'
        return jsonify({'status': 'fail', 'message': e, 'code': 500})

    limit = int(request.args.get('limit', 10))
    neighbors = app.semantic_neighbors.query(query, limit)
    if neighbors:
        neighbors = [{'word': w, 'sim': d} for w, d in neighbors]
    return jsonify({'status': 'OK', 'results': neighbors})


@app.route('/api/phrase', methods=['GET'])
@swag_from('../openapi/phrase.yml')
def phrase_neighbors(limit=10):
    if 'q' in request.args and request.args['q'].strip():
        query = request.args['q'].strip()
    else:
        e = 'Error: No q-field provided. Please specify a non-empty phrase.'
        return jsonify({'status': 'fail', 'message': e, 'code': 500})

    limit = int(request.args.get('limit', limit))
    neighbors = app.sentence_neighbors.query(query, limit)
    if neighbors:
        neighbors = [
            {
                'sentence': Text.query.get(doc_id + 1).get(sent_id),
                'distance': str(dist)
            }
            for (doc_id, sent_id), dist in neighbors
        ]
    return jsonify({'status': 'OK', 'results': neighbors})


@app.route('/api/concordance', methods=['GET'])
@swag_from('../openapi/concordance.yml')
def concordance(limit=5):    
    if 'q' in request.args and request.args['q'].strip():
        query = request.args['q'].strip()
    else:
        e = 'Error: No w-field provided. Please specify a non-empty word.'
        return jsonify({'status': 'fail', 'message': e, 'code': 500})

    limit = int(request.args.get('limit', limit))

    hits, snippets, total = query_index('echoes-texts', query, limit=limit)
    return jsonify({'hits': hits, 'snippets': snippets, 'total': total})


@app.route('/api/complete', methods=['GET'])
@swag_from('../openapi/complete.yml') # todo
def complete(temp=.35, limit=5, length=60):    
    if 'q' in request.args and request.args['q']:
        query = request.args['q']
    else:
        query = ''

    limit = int(request.args.get('limit', 5))
    length = int(request.args.get('length', length))
    temp = float(request.args.get('temp', temp))

    hypotheses = app.completer.query(query, num_alternatives=limit,
                                     sugg_len=length, temp=temp)
    return jsonify({'hypotheses': hypotheses})

