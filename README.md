# Echoes

Some introductory sentences about the project, its main purpose and history. 
<img src="images/Echo.jpg" align="right" height="250" />

## Installation

The system requires `elasticsearch` to be installed, which is likely available in most
recent Linux distributions. On macOS, install `elasticsearch` with homebrew using:

```
brew install elasticsearch
```

To install all required packages and create the echoes databases, indexes and embeddings,
run: 

```
make
```

Other make options are:
1. `make clean`: to clear out the database and elasticsearch index;
2. `make install`: to install all Python dependencies;
3. `make index`: to build the databases, indexes, and embedding files
