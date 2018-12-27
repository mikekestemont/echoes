from api import app


def index_document(index, model):
    payload = {}
    for field in model.__searchable__:
        payload[field] = getattr(model, field)
    app.elasticsearch.index(index=index, doc_type=index, id=model.id, body=payload)

def query_index(index, query):
    search = app.elasticsearch.search(
        index=index, doc_type=index,
        body={'query': {'multi_match': {'query': query, 'fields': ['*']}},
              'highlight': {'fields': {'text': {}}}})
    ids, snippets = zip(*[(int(hit['_id']), hit['highlight']['text'])
                          for hit in search['hits']['hits']])
    return ids, snippets, search['hits']['total']
