# Word2doc2vec-Doc-relevance
This repository focuses on an approach exploring and assessing literature-based doc-2-doc recommendations using the Word2Vec technique, followed  centroid aggregation method to create document-level embeddings. The approach is applied to the RELISH dataset.

# Approach
Our approach involves utilizing Word2Vec for capturing word-level semantics and generating word embeddings. We then employ the centroid approach to create document-level embeddings, which entails calculating the centroids of word embeddings within each document's title and abstract. For word embeddings, we utilize both pretrained and trained models.

By default, we use the "word2vec-google-news-300" Gensim word2vec model as the pretrained model for word embeddings. However, it can be substituted with other pretrained models available in the Gensim library.

For our trained model, we utilize the documents from the RELISH dataset as the training corpus. The words from the titles and abstracts of each document are combined and input into the model as a single document.

# Input Data
The input data for this method consists of preprocessed tokens derived from the RELISH documents. These tokens are stored in the RELISH.npy file, which contains preprocessed arrays comprising PMIDs, document titles, and abstracts. These arrays are generated through an extensive preprocessing pipeline, as elaborated in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: structural words are eliminated, text is converted to lowercase, and finally, tokenization is employed, resulting in arrays of individual words.

# Generating Embeddings
The following section outlines the process of generating document-level embeddings for each PMID of the RELISH corpus.

## Word embeddings

### Utilizing Trained Word2Vec models
We construct Word2Vec models with customizable hyperparameters. We employ the parameters shown below in order to generate our models.
#### Parameters

+ **dm:** {1,0} Refers to the training algorithm. If dm=1, distributed memory is used otherwise, distributed bag of words is used.
+ **vector_size:** It represents the number of dimensions our embeddings will have.
+ **window:** It represents the maximum distance between the current and predicted word.
+ **epochs:** It is the nuber of iterations of the training dataset.
+ **min_count:** It is the minimum number of appearances a word must have to not be ignored by the algorithm.

### Utilizing Pre-trained Word2Vec models
By default, we make use of the Gensim Word2Vec model "word2vec-google-news-300" to generate pre-trained word embeddings.

## Document Embeddings
Document embeddings are created by computing the centroids of all provided word embeddings within each title and abstract document. The resulting embeddings generated from various model hyperparameter configurations are stored. These embeddings, along with their respective PMIDs, are saved as a dataframe in a pickle file.

# Hyperparameter Optimization
*To be written*

# Code Implementation
The `generate_embeddings.py` script uses the RELISH Tokenized npy file as input and supports the generation and training of Word2Vec models, generation of embeddings and saving the embeddings as pickle files. The script includes a default parameter dictionarry with present hyperparameters. YOu can easily adapt it for different values and parameters by modifying the `params` dictionary.

The script consists of two main functions `generateWord2VecModel` and `generateDocumentEmbeddings`.

`generateWord2VecModel` : This function creates a Word2vec model using the provided sentences and the inputer hyper parameter.

`generateDocumentEmbeddings` :  This function generates document embeddings from titles and abstracts using Word2Vec and centroid calculations. It can utilize either an existing or a default pretrained model for word embeddings and saves the results as .npy or .pkl files based on the specified format.

# Code Execution

To run this script, please execute the following command:

`python3 code/generate_embeddings.py --input data/RELISH_tokenized.npy --output data/ --params_json data/hyperparameters_word2vec --use_pretrained 0`

# Tutorial
A [tutorial](./docs/Tutorial.ipynb) is accessible in the form of Jupyter notebook for the generation of embeddings.