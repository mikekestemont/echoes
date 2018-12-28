all: clean install build

clean:
	echo "Cleaning up db and index..."
	@if [ -e echoes.db ]; then rm echoes.db; fi;
	@curl -XDELETE 'http://localhost:9200/echoes-texts/'
	@if [ -e __pycache__ ]; then rm -r __pycache__; fi;

install:
	echo "Installing packages from requirements.txt..."
	@pip install -r requirements.txt

build:
	echo "Preprocessing corpus..."
	@python echoes/preprocess_corpus.py
	echo "Constructing word embeddings..."
	@python echoes/word_embeddings.py
	echo "Building SQLite index and filling elasticsearch index..."
	@python echoes/build_index.py
