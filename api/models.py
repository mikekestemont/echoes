from api import app, db


class Text(db.Model):
    __searchable__ = ['text']
    
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String())
    author = db.Column(db.String())
    text = db.Column(db.String())

    def get_sentence(self, i):
        return self.text.split('\n')[i]
