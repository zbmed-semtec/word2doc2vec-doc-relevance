# word2doc2vec-doc-relevance
An approach exploring and assessing literature-based doc-2-doc recommendations using word2vec combined with doc2vec, and applying it to TREC and RELISH datasets

# word2vec embeddings
We used the gensim word2vec model [glove-wiki-gigaword-200](https://nlp.stanford.edu/projects/glove/) as the default model for pre-trained word embeddings.
This can also be replaced by your own model given two .txt files, one of which contains line separated terms for the plaintext words and the other line separated word vectors,
where each dimension is separated by space.