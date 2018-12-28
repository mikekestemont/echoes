import argparse
import glob
import os

from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.moses import MosesDetokenizer
import numpy as np
from elmoformanylangs import Embedder

import faiss

import sqlite3


def main():
    parser = argparse.ArgumentParser(description='Sets up databases')
    parser.add_argument('--reparse', default=False, action='store_true',
                        help='Reparse the database from scratch')
    args = parser.parse_args()
    print(args)

    db_filename = 'sentence_str.db'

    e = Embedder('nl')
    faiss_db = faiss.IndexFlatL2(1024)
    detokenizer = MosesDetokenizer()

    if args.reparse:
        # create database:
        try:
            os.remove(db_filename)
        except FileNotFoundError:
            pass

        conn = sqlite3.connect(db_filename)
        c = conn.cursor()
        c.execute('''CREATE TABLE sentences(sent_id INTEGER PRIMARY KEY,
                                            sentence text)''')
        
        for fn in sorted(list(glob.glob('my_corpus/*.txt'))[:1]):
            cnt = 0
            print(fn)
            sentences = []
            with open(fn) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cnt += 1
                    if cnt >= 1000:
                        break
                    for s in sent_tokenize(line):
                        sentences.append(word_tokenize(s))
                         # specify NULL as primary key, to ensure autoincrement:
                        c.execute("INSERT INTO sentences VALUES(NULL, ?)", (s, ))
                        
            conn.commit()
            
            X = e.sents2elmo(sentences)
            X = np.array([x.mean(axis=0) for x in X])    
            faiss_db.add(X)
            print(f'-> Currently indexed {faiss_db.ntotal} sentences')

        conn.close()
        faiss.write_index(faiss_db, 'sentence_vec.db')
    
    faiss_db = faiss.read_index('sentence_vec.db')
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()

    test_sentences = []
    with open('test.txt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for s in sent_tokenize(line):
                test_sentences.append(word_tokenize(s))
            
    test_sentences = test_sentences[:20]
    X = e.sents2elmo(test_sentences)
    X = np.array([x.mean(axis=0) for x in X])    
    
    distances, indices = faiss_db.search(X, k=6)

    for idx in range(distances.shape[0]):
        print('query:', detokenizer.detokenize(test_sentences[idx], return_str=True))
        
        print('neighbors:')
        for d, i in zip(distances[idx], indices[idx]):
            sent = c.execute("SELECT sentence FROM sentences WHERE rowid = ?", (int(i), )).fetchone()[0]
            print('   - ', sent, d)
        
        print('=' * 64)

if __name__ == '__main__':
    main()