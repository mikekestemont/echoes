import unidecode

from . import app


def cleanup_snippet(snippet):
    return ' '.join(unidecode.unidecode(snippet).split())


def index_document(index, model):
    payload = {}
    for field in model.__searchable__:
        contents = getattr(model, field)
        if isinstance(contents, (tuple, list)):
            contents = '\n'.join(contents)
        payload[field] = contents
    app.elasticsearch.index(index=index, doc_type=index, id=model.id, body=payload)


def query_index(index, query, limit=5):
    search = app.elasticsearch.search(
        index=index, doc_type=index,
        body={
            "from" : 0, "size" : limit,
            'query': {'match': {'text': query}},
            'highlight': {
                'fields': {
                    'text': {
                        "fragment_size": app.config['FRAGMENT_SIZE'],
                        "number_of_fragments": app.config['NUMBER_OF_FRAGMENTS']
                    }}}
        })
    ids, snippets = [], []
    if search['hits']['hits']:
        ids, snippets = zip(*[(int(hit['_id']), hit['highlight']['text'])
                              for hit in search['hits']['hits']])
        snippets = [[cleanup_snippet(s) for s in snippet] for snippet in snippets]
    return ids, snippets, search['hits']['total']
