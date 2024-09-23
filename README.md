[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

[![SWH](https://archive.softwareheritage.org/badge/swh:1:dir:51e827e63084faece4fe491da89c9ffb0ef9a92c/)](https://archive.softwareheritage.org/swh:1:dir:51e827e63084faece4fe491da89c9ffb0ef9a92c;origin=https://github.com/zbmed-semtec/word2doc2vec-doc-relevance;visit=swh:1:snp:028e1ef544c731738cbfa4c3ba073522bf53987e;anchor=swh:1:rev:248bd9af17a7c09df3572fc854e714e00cad51e5)

# Word2doc2vec-Doc-relevance
This repository focuses on an approach exploring and assessing literature-based doc-2-doc recommendations using the Word2Vec technique, followed  centroid aggregation method to create document-level embeddings. The approach is applied to the RELISH dataset.

## Table of Contents

1. [About](#about)
2. [Input Data](#input-data)
3. [Pipeline](#pipeline)
    1. [Generate Embeddings](#generate-embeddings)
        - [Using Trained Word2Vec models](#using-trained-word2vec-models)
          - [Parameters](#parameters)
          - [Hyperparameters](#hyperparameters)
        - [Using Pre-trained Word2Vec models](#using-pre-trained-word2vec-models)
        - [Document Embeddings](#document-embeddings)
    2. [Calculate Cosine Similarity](#calculate-cosine-similarity)
    3. [Evaluation](#evaluation)
        - [Precision@N](#precisionn)
        - [nDCG@N](#ndcgn)
4. [Code Implementation](#code-implementation)
5. [Getting Started](#getting-started)
6. [Tutorial](#tutorial)

## About

Our approach involves utilizing [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf) for capturing word-level semantics and generating word embeddings. We then employ the centroid approach to create document-level embeddings, which entails calculating the centroids of word embeddings within each document's title and abstract. For word embeddings, we utilize both pretrained and trained models.

## Input Data

The input data for this method consists of preprocessed tokens derived from the RELISH documents. These tokens are stored in the RELISH.npy file, which contains preprocessed arrays comprising PMIDs, document titles, and abstracts. These arrays are generated through an extensive preprocessing pipeline, as elaborated in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: structural words are eliminated, text is converted to lowercase, and finally, tokenization is employed, resulting in arrays of individual words.

## Pipeline

The following section outlines the process of generating document-level embeddings through hyperparameter optimization, computing the cosine similarity scores and evaluating the given similarity results with the relevance matrix.

### Generate Embeddings
The following section outlines the process of generating document-level embeddings out of word-level embeddings for each PMID of the RELISH corpus.

#### Using Trained Word2Vec models
We construct Word2Vec models with customizable hyperparameters. We employ the parameters shown below in order to generate our models.
##### Parameters

+ **sg:** {1,0} Refers to the training algorithm. If sg=1, skim grams is used otherwise, continuous bag of words (CBOW) is used.
+ **vector_size:** It represents the number of dimensions our embeddings will have.
+ **window:** It represents the maximum distance between the current and predicted word.
+ **epochs:** It is the nuber of iterations of the training dataset.
+ **min_count:** It is the minimum number of appearances a word must have to not be ignored by the algorithm.

#### Hyperparameters
The hyperparameters can be modified in [`hyperparameters.yaml`](./code/hyperparameters.yaml)


#### Document Embeddings
Document embeddings are created by computing the centroids of all provided word embeddings within each title and abstract document. The resulting embeddings generated from various model hyperparameter configurations are stored. These embeddings, along with their respective PMIDs, are saved as a dataframe in a pickle file. Each specific set of hyperparameter combination results in having a separate pickle file.

### Calculate Cosine Similarity

To assess the similarity between two documents within the RELISH corpus, we employ the Cosine Similarity metric. This process enables the generation of a 4-column matrix containing cosine similarity scores for existing pairs of PMIDs within our corpus. For a more detailed explanation of the process, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Cosine_Similarity).

## Evaluation

### Precision@N

In order to evaluate the effectiveness of this approach, we make use of Precision@N. Precision@N measures the precision of retrieved documents at various cutoff points (N).We generate a Precision@N matrix for existing pairs of documents within the RELISH corpus, based on the original RELISH JSON file. The code determines the number of true positives within the top N pairs and computes Precision@N scores. The result is a Precision@N matrix with values at different cutoff points, including average scores. For detailed insights into the algorithm, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Precision%40N_existing_pairs).

### nDCG@N

Another metric used is the nDCG@N (normalized Discounted Cumulative Gain). This ranking metric assesses document retrieval quality by considering both relevance and document ranking. It operates by using a TSV file containing relevance and cosine similarity scores, involving the computation of DCG@N and iDCG@N scores. The result is an nDCG@N matrix for various cutoff values (N) and each PMID in the corpus, with detailed information available in the [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation).

## Code Implementation

The [`generate_embeddings.py`](./code/generate_embeddings.py) script uses the RELISH Tokenized npy file as input and supports the generation and training of Word2Vec models, generation of embeddings and saving the embeddings as pickle files.

The script consists of two main functions `generateWord2VecModel` and `generateDocumentEmbeddings`.

`generateWord2VecModel` : This function creates a Word2vec model using the provided sentences and the inputer hyper parameter.

`generateDocumentEmbeddings` :  This function generates document embeddings from titles and abstracts using Word2Vec and centroid calculations. It can utilize either an existing or a default pretrained model for word embeddings and saves the results as .npy or .pkl files based on the specified format.

## Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using the following command:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/word2doc2vec-doc-relevance.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/word2doc2vec-doc-relevance.git
```

### Step 2: Create a virtual environment and install dependencies

To create a virtual environment within your repository, run the following command:

```
python3 -m venv .venv 
source .venv/bin/activate   # On Windows, use '.venv\Scripts\activate' 
```

To confirm if the virtual environment is activated and check the location of yourPython interpreter, run the following command:

```
which python    # On Windows command prompt, use 'where python'
                # On Windows PowerShell, use 'Get-Command python'
```
The code is stable with python 3.6 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```
### Step 3: Dataset

Use the Download_Dataset.sh script to download the Split Dataset by 
running the following commands:

```
chmod +777 Download_Dataset.sh
./Download_Dataset.sh
```

This script makes sure that the necessary folders are created and the files are downloaded in the corresponding folders as shown below.

```
📦 /word2doc2vec-doc-relevance
└─ data
   └─ Input
      ├─ Tokens
      │  ├─ relish.npy
      └─ Ground_truth
         └─ relevance_matrix.tsv
```

The file *relish.npy* is in the NumPy binary format (.npy), which is specifically used to store NumPy arrays efficiently. These arrays contain the PMID, title, and abstract for each document.

In contrast, *relevance_matrix.tsv* is a Tab-separated Values file, similar to CSV but using tabs as delimiters. It stores tabular data with four columns: PMID1 | PMID2 | Relevance | WMD Similarity.

Reference: Tab-separated values (TSV) file format:  
[![FAIRsharing DOI](https://img.shields.io/badge/DOI-10.25504%2FFAIRsharing.a978c9-blue)](https://doi.org/10.25504/FAIRsharing.a978c9)

### Step 4: Generate Embeddings
The [`generate_embeddings.py`](./code/generate_embeddings.py) script uses the RELISH Tokenized npy file as input. You can easily adapt it for different values and parameters by modifying the [[`hyperparameters.yaml`](./code/hyperparameters.yaml) Make sure to have the RELISH Tokenized.npy file within the directory under the data folder.
```
python3 code/generate_embeddings.py [-i INPUT PATH] [-o OUTPUT PATH] [-p PARAMS]
```

You must pass the following arguments:

+ -i/ --input : File path to the RELISH tokenized .npy file.
+ -o/ --output : File path to the resulting embeddings in pickle file format.
+ -p/ --params : File path to the hyperparameters YAML file.

To run this script, please execute the following command:

```
python3 code/generate_embeddings.py --input data/Input/Tokens/relish.npy --output data/embeddings --params code/hyperparameters.yaml 
```

The script will create document embeddings, and store them in separate directories. You should expect to find a total of 18 files corresponding to the various models, embeddings, and embedding pickle files.

### Step 5: Calculate Cosine Similarity
In order to generate the cosine similarity matrix and execute this [script](./code/generate_cosine_existing_pairs.py), run the following command:

```
python3 code/generate_cosine_existing_pairs.py [-i INPUT PATH] [-e EMBEDDINGS] [-o OUTPUT PATH]
```

You must pass the following four arguments:

+ -i/ --input : File path to the RELISH relevance matrix in the TSV format.
+ -e/ --embeddings : File path to the embeddings in the pickle file format.
+ -o/ --output : File path for the output 4 column cosine similarity matrix.


For example, if you are running the code from the code folder and have the RELISH relevance matrix in the data folder, run the cosine matrix creation for all hyperparameters as:

```
python3 code/generate_cosine_existing_pairs.py -i data/Input/Ground_truth/relevance_matrix.tsv -e data/embeddings/embeddings_pickle.pkl -o data/cosine/cosine_similarity.tsv
```

Note: You would have to run the above command for every hyperparameter configuration by changing the file name for the embedding's pickle file or use the following shell script to generate all files at once.

```
for VALUE in {0..17};do
python3 code/generate_cosine_existing_pairs.py -i data/Input/Ground_truth/relevance_matrix.tsv -e data/embeddings/embeddings_pickle_${VALUE}.pkl -o data/cosine/cosine_similarity_${VALUE}.tsv
done
```

### Step 6: Precision@N
In order to calculate the Precision@N scores and execute this [script](/code/precision.py), run the following command:

```
python3 code/precision.py [-i COSINE FILE PATH]  [-o OUTPUT PATH] [-c CLASSES]
```

You must pass the following two arguments:

+ -i/ --cosine_file_path: path to the 4-column cosine similarity existing pairs RELISH file: (tsv file)
+ -o/ --output_path: path to save the generated precision matrix: (tsv file)
+ -c/ --classes: Number of classes for class distribution (2 or 3)

For example, if you are running the code from the code folder and have the cosine similarity TSV file in the data folder, run the precision matrix creation for the first hyperparameter as:

```
python3 code/precision.py -i data/cosine_similarity_0.tsv -o data/precision_three_classes/precision_0.tsv -c 3
```

Note: You would have to run the above command for every hyperparameter configuration by changing the file name for the cosine similarity file or use the following shell script to generate all files at once.


```
for VALUE in {0..17};do
python3 code/precision.py -c data/cosine_similarity_${VALUE}.tsv -o data/precision_three_classes/precision_${VALUE}.tsv -c 3
done
```

Note: Make sure to re-run the above command by changing the classes for a different class distribution.

### Step 7: nDCG@N
In order to calculate nDCG scores and execute this [script](/code/calculate_gain.py), run the following command:

```
python3 code/calculate_gain.py [-i INPUT]  [-o OUTPUT]
```

You must pass the following two arguments:

+ -i / --input: Path to the 4 column cosine similarity existing pairs RELISH TSV file.
+ -o/ --output: Output path along with the name of the file to save the generated nDCG@N TSV file.

For example, if you are running the code from the code folder and have the 4 column RELISH TSV file in the data folder, run the matrix creation for the first hyperparameter as:

```
python3 code/calculate_gain.py -i data/cosine/cosine_similarity_0.tsv -o data/gain/ndcg_0.tsv
```

Note: You would have to run the above command for every hyperparameter configuration by changing the file name for the cosine similarity file or use the following shell script to generate all files at once.

```
for VALUE in {0..17};do
python3 code/calculate_gain.py -i data/cosine/cosine_similarity_${VALUE}.tsv -o data/gain/ndcg_${VALUE}.tsv
done
```

### Step 8: Compile Results

In order to compile the average result values for Precison@ and nDCG@N and generate a single TSV file each, please use this [script](code/show_avg.py).

You must pass the following two arguments:

+ -i / --input: Path to the directory consisting of all the precision matrices/gain matrices.
+ -o/ --output: Output path along with the name of the file to save the generated compiled Precision@N / nDCG@N TSV file.


If you are running the code from the code folder, run the compilation script as:

```
python3 code/show_avg.py -i data/gain/ -o data/results_gain.tsv
```

NOTE: Please do not forget to put a `'/'` at the end of the input file path.


## Tutorial
A [tutorial](./docs/Tutorial.ipynb) is accessible in the form of Jupyter notebook for the generation of embeddings.
