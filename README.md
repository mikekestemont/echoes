# Echoes

Intro text

![Echo by Alexandre Cabanel. Painted in 1874 the piece now hangs in the Metropolitan
Museum of Art, New York.](images/Echo.jpg =250x)

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
3. `make build`: to build te databases, indexes, and embedding files
