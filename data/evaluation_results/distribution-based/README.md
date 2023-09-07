# Evaluation of the word2doc2vec-doc-relevance with the distribution-based approach

The following tables show the results of the [distribution-based](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Distribution_Analysis) evaluation approach, when applied on to "word2doc2vec-doc-relevance" technique. 
These results are calculated for the different hyper-parameters settings of the Word2Vec approach to obtain the optimal combination for each dataset used in this work.

Each of the below tables contains seven columns. The first six of these columns represent the hyper-parameters of the Word2Vec model:
- **sg:** Defines the training algorithm that is used. If sg=0, 'continuous bag of words' (CBOW) is used, else if sg=1, 'skip-gram' is used.
- **epochs:** Defines the number of iterations over the corpus.
- **min_count:** Ignores those words that have the total frequency less than this number.
- **vector_size:** Defines the dimensionality of the feature vector.
- **window:** Defines the maximum distance between the current and predicted word within a sentence.
- **workers:** Defines the working threads used to train the model.
- **AUC:** Defines the "area under the curve", as a result of this evaluation approach.

**RELISH:** The table below shows the results from the distribution-based approach for the RELISH dataset.

| sg    | epochs      | min_count     | vector_size | window  | workers | AUC    |
|:-----:|:-----------:|:-------------:|:-----------:|:-------:|:-------:|:------:|
| 0     | 15          | 5             | 200         | 5       | 8       | 0.5595 |
| 0     | 15          | 5             | 200         | 6       | 8       | 0.561  |
| 0     | 15          | 5             | 200         | 7       | 8       | 0.5624 |
| 0     | 15          | 5             | 300         | 5       | 8       | 0.5597 |
| 0     | 15          | 5             | 300         | 6       | 8       | 0.5624 |
| 0     | 15          | 5             | 300         | 7       | 8       | 0.5639 |
| 0     | 15          | 5             | 400         | 5       | 8       | 0.5609 |
| 0     | 15          | 5             | 400         | 6       | 8       | 0.5625 |
| 0     | 15          | 5             | 400         | 7       | 8       | 0.5649 |
| 1     | 15          | 5             | 200         | 5       | 8       | 0.5945 |
| 1     | 15          | 5             | 200         | 6       | 8       | 0.5969 |
| 1     | 15          | 5             | 200         | 7       | 8       | 0.5991 |
| 1     | 15          | 5             | 300         | 5       | 8       | 0.5982 |
| 1     | 15          | 5             | 300         | 6       | 8       | 0.6005 |
| 1     | 15          | 5             | 300         | 7       | 8       | 0.603  |
| 1     | 15          | 5             | 400         | 5       | 8       | 0.5993 |
| 1     | 15          | 5             | 400         | 6       | 8       | 0.6024 |
| 1     | 15          | 5             | 400         | 7       | 8       | 0.6046 |

**TREC-simplified:** The table below shows the results from the distribution-based approach for the "TREC-simplified" variant of the TREC dataset.

| sg    | epochs      | min_count     | vector_size | window  | workers | AUC    |
|:-----:|:-----------:|:-------------:|:-----------:|:-------:|:-------:|:------:|
| 0     | 15          | 5             | 200         | 5       | 8       | 0.6283 |
| 0     | 15          | 5             | 200         | 6       | 8       | 0.627  |
| 0     | 15          | 5             | 200         | 7       | 8       | 0.6253 |
| 0     | 15          | 5             | 300         | 5       | 8       | 0.6275 |
| 0     | 15          | 5             | 300         | 6       | 8       | 0.6269 |
| 0     | 15          | 5             | 300         | 7       | 8       | 0.6254 |
| 0     | 15          | 5             | 400         | 5       | 8       | 0.628  |
| 0     | 15          | 5             | 400         | 6       | 8       | 0.6268 |
| 0     | 15          | 5             | 400         | 7       | 8       | 0.6266 |
| 1     | 15          | 5             | 200         | 5       | 8       | 0.653  |
| 1     | 15          | 5             | 200         | 6       | 8       | 0.6557 |
| 1     | 15          | 5             | 200         | 7       | 8       | 0.6576 |
| 1     | 15          | 5             | 300         | 5       | 8       | 0.6542 |
| 1     | 15          | 5             | 300         | 6       | 8       | 0.6568 |
| 1     | 15          | 5             | 300         | 7       | 8       | 0.6593 |
| 1     | 15          | 5             | 400         | 5       | 8       | 0.6547 |
| 1     | 15          | 5             | 400         | 6       | 8       | 0.6576 |
| 1     | 15          | 5             | 400         | 7       | 8       | 0.6595 |

**TREC-repurposed:** The table below shows the results from the distribution-based approach for the "TREC-repurposed" variant of the TREC dataset.

| sg    | epochs      | min_count     | vector_size | window  | workers | AUC    |
|:-----:|:-----------:|:-------------:|:-----------:|:-------:|:-------:|:------:|
| 0     | 15          | 5             | 200         | 5       | 8       | 0.7158 |
| 0     | 15          | 5             | 200         | 6       | 8       | 0.7125 |
| 0     | 15          | 5             | 200         | 7       | 8       | 0.7125 |
| 0     | 15          | 5             | 300         | 5       | 8       | 0.7176 |
| 0     | 15          | 5             | 300         | 6       | 8       | 0.7126 |
| 0     | 15          | 5             | 300         | 7       | 8       | 0.7101 |
| 0     | 15          | 5             | 400         | 5       | 8       | 0.7151 |
| 0     | 15          | 5             | 400         | 6       | 8       | 0.7135 |
| 0     | 15          | 5             | 400         | 7       | 8       | 0.7128 |
| 1     | 15          | 5             | 200         | 5       | 8       | 0.7627 |
| 1     | 15          | 5             | 200         | 6       | 8       | 0.7671 |
| 1     | 15          | 5             | 200         | 7       | 8       | 0.7697 |
| 1     | 15          | 5             | 300         | 5       | 8       | 0.7649 |
| 1     | 15          | 5             | 300         | 6       | 8       | 0.7689 |
| 1     | 15          | 5             | 300         | 7       | 8       | 0.7722 |
| 1     | 15          | 5             | 400         | 5       | 8       | 0.7653 |
| 1     | 15          | 5             | 400         | 6       | 8       | 0.7698 |
| 1     | 15          | 5             | 400         | 7       | 8       | 0.7733 |

