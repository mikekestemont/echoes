from flask import request, jsonify

from .search import query_index

from api import app


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
        return jsonify({'status': 'fail', 'message': e, 'code': 500})
        
    topn = int(request.args.get('topn', topn))
    neighbors = app.semantic_neighbors.query(w, topn)
    if neighbors:
        neighbors = [{'word': w, 'sim': d} for w, d in neighbors]
    return jsonify({'status': 'OK', 'results': neighbors})


@app.route('/api/concordance', methods=['GET'])
def concordance(limit=20):
    if 'w' in request.args and request.args['w'].strip():
        w = request.args['w'].strip()
    else:
        e = 'Error: No w-field provided. Please specify a non-empty word.'
        return jsonify({'status': 'fail', 'message': e, 'code': 500})

    topn = int(request.args.get('topn', limit))

    json_out = []
    hits, total = query_index('echoes-texts', w)
    return jsonify({'hits': hits, 'total': total})
