from . import app, db
import json


class JSONEncodedDict(db.TypeDecorator):
    "Represents an immutable structure as a json-encoded string."

    impl = db.VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class Text(db.Model):
    __searchable__ = ['text']

    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String())
    author = db.Column(db.String())
    text = db.Column(JSONEncodedDict)

    def get(self, i):
        return self.text[i]
