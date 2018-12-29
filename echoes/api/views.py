from flask import request, jsonify

from .search import query_index
from .models import Text

from . import app


@app.route('/api/word', methods=['GET'])
def word_neighbors():
    """Retrieve a list of synonyms for the given word.
    ---
    parameters:
      - name: q
        in: query
        type: string
        required: true
    responses:
      200:
        description: A list of neighboring words
    """

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
def phrase_neighbors(limit=10):
    """Retrieve a list of phrase synonyms for the given query.
    ---
    parameters:
      - name: q
        in: query
        type: string
        required: true
    responses:
      200:
        description: A list of neighboring phrases.
    """
    
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
def concordance(limit=5):
    """Retrieve a list of related sentences for the given query.
    ---
    parameters:
      - name: q
        in: query
        type: string
        required: true
    responses:
      200:
        description: A list of related sentences.
    """
    
    if 'q' in request.args and request.args['q'].strip():
        query = request.args['q'].strip()
    else:
        e = 'Error: No w-field provided. Please specify a non-empty word.'
        return jsonify({'status': 'fail', 'message': e, 'code': 500})

    limit = int(request.args.get('limit', limit))

    hits, snippets, total = query_index('echoes-texts', query, limit=limit)
    return jsonify({'hits': hits, 'snippets': snippets, 'total': total})


@app.route('/api/sentence', methods=['GET'])
def sentence():
    document_id = int(request.args['did'])
    sentence_id = int(request.args['sid'])
    sentence = Text.query.get(document_id)
    return jsonify({'sentence': sentence.get(sentence_id)})
