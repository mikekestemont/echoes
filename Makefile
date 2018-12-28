all: clean install index train

clean:
	echo "Cleaning up db and index..."
	@if [ -e data/echoes.db ]; then rm data/echoes.db; fi;
	@curl -XDELETE 'http://localhost:9200/echoes-texts/'
	@find . -name "*.py[co]" -o -name __pycache__ -exec rm -rf {} +

install:
	echo "Installing packages from requirements.txt..."
	@pip install -r requirements.txt

train:
	echo "Constructing word embeddings..."
	@python echoes/word_embeddings.py

index:
	@if ! (curl -s http://localhost:9200/ | grep "elasticsearch" > /dev/null); \
	then (echo "Elastic search needs to be running on port 9200"; exit 1); fi;
	echo "Preprocessing corpus..."
	@python echoes/preprocess_corpus.py
	echo "Building SQLite index and filling elasticsearch index..."
	@python echoes/build_index.py
