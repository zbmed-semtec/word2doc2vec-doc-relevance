# Evaluation of the word2doc2vec-doc-relevance using nDCG@N approach

The following tables show the results of the [nDCG@N](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation) evaluation approach, when applied on to "word2doc2vec-doc-relevance" technique.
These results are calculated for the different hyper-parameters settings of the Word2Vec approach to obtain the optimal combination for each dataset used in this work.

Each of the below tables contains seven columns. The first six of these columns represent the hyper-parameters of the Word2Vec model. The remaining six columns represent the average nDCG scores at different values of N:
- **sg:** Defines the training algorithm that is used. If sg=0, 'continuous bag of words' (CBOW) is used, else if sg=1, 'skip-gram' is used.
- **epochs:** Defines the number of iterations over the corpus.
- **min_count:** Ignores those words that have the total frequency less than this number.
- **vector_size:** Defines the dimensionality of the feature vector.
- **window:** Defines the maximum distance between the current and predicted word within a sentence.
- **workers:** Defines the working threads used to train the model.
- **nDCG@5 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 5 articles retrieved.
- **nDCG@10 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 10 articles retrieved.
- **nDCG@15 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 15 articles retrieved.
- **nDCG@20 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 20 articles retrieved.
- **nDCG@25 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 25 articles retrieved.
- **nDCG@50 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 50 articles retrieved.

**RELISH:** The table below shows the results from the nDCG@N approach for the RELISH dataset.

| sg  | epochs  | min_count  | vector_size | window  | workers | nDCG@5 (AVG) | nDCG@10 (AVG) | nDCG@15 (AVG) | nDCG@20 (AVG) | nDCG@25 (AVG) | nDCG@50 (AVG) |
|:---:|:-------:|:----------:|:-----------:|:-------:|:-------:|:------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| 0   | 15      | 5          | 200         | 5       | 8       | 0.6545       |	0.6379        |	0.6363        |	0.645         |	0.6594        |	0.7712        |
| 0   | 15      | 5          | 200         | 6       | 8       | 0.6531       |	0.6372        |	0.6361        |	0.645         |	0.6585        |	0.7714        |
| 0   | 15      | 5          | 200         | 7       | 8       | 0.6529       |	0.6374        |	0.6364        |	0.6449        |	0.659         |	0.7714        |
| 0   | 15      | 5          | 300         | 5       | 8       | 0.6557       |	0.6367        |	0.637         |	0.6453        |	0.6593        |	0.7713        |
| 0   | 15      | 5          | 300         | 6       | 8       | 0.6542       |	0.6369        |	0.6361        |	0.6453        |	0.6588        |	0.7713        |
| 0   | 15      | 5          | 300         | 7       | 8       | 0.6561       |	0.638         |	0.6371        |	0.6455        |	0.6596        |	0.7719        |
| 0   | 15      | 5          | 400         | 5       | 8       | 0.6558       |	0.638         |	0.6372        |	0.6458        |	0.6594        |	0.772         |
| 0   | 15      | 5          | 400         | 6       | 8       | 0.6553       |	0.6371        |	0.6371        |	0.6453        |	0.6591        |	0.7716        |
| 0   | 15      | 5          | 400         | 7       | 8       | 0.6543       |	0.6376        |	0.6365        |	0.6454        |	0.6592        |	0.7715        |
| 1   | 15      | 5          | 200         | 5       | 8       | 0.7826       |	0.7513        |	0.7415        |	0.7419        |	0.7483        |	0.8333        |
| 1   | 15      | 5          | 200         | 6       | 8       | 0.7832       |	0.7513        |	0.7422        |	0.7428        |	0.7496        |	0.8338        |
| 1   | 15      | 5          | 200         | 7       | 8       | 0.7838       |	0.752         |	0.7422        |	0.7429        |	0.7494        |	0.8342        |
| 1   | 15      | 5          | 300         | 5       | 8       | 0.7834       |	0.75          |	0.7421        |	0.7427        |	0.7503        |	0.8334        |
| 1   | 15      | 5          | 300         | 6       | 8       | 0.7834       |	0.7505        |	0.7425        |	0.7434        |	0.7499        |	0.8337        |
| 1   | 15      | 5          | 300         | 7       | 8       | 0.7839       |	0.7514        |	0.7429        |	0.7436        |	0.7508        |	0.8345        |
| 1   | 15      | 5          | 400         | 5       | 8       | 0.7801       |	0.7496        |	0.7416        |	0.7415        |	0.7483        |	0.8333        |
| 1   | 15      | 5          | 400         | 6       | 8       | 0.7799       |	0.7509        |	0.7423        |	0.7411        |	0.7489        |	0.8341        |
| 1   | 15      | 5          | 400         | 7       | 8       | 0.7816       |	0.7521        |	0.7428        |	0.7426        |	0.7504        |	0.8346        |

**TREC-repurposed:** The table below shows the results from the nDCG@N approach for the "TREC-repurposed" variant of the TREC dataset.


| sg  | epochs  | min_count  | vector_size | window  | workers | nDCG@5 (AVG) | nDCG@10 (AVG) | nDCG@15 (AVG) | nDCG@20 (AVG) | nDCG@25 (AVG) | nDCG@50 (AVG) |
|:---:|:-------:|:----------:|:-----------:|:-------:|:-------:|:------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| 0   | 15      | 5          | 200         | 5       | 8       | 0.4955       |	0.4874        |	0.4843        |	0.4829        |	0.4826        |	0.4902        |
| 0   | 15      | 5          | 200         | 6       | 8       | 0.4958       |	0.4882        |	0.4846        |	0.4825        |	0.4829        |	0.4908        |
| 0   | 15      | 5          | 200         | 7       | 8       | 0.4981       |	0.4907        |	0.4862        |	0.485         |	0.4848        |	0.4925        |
| 0   | 15      | 5          | 300         | 5       | 8       | 0.4958       |	0.4892        |	0.4856        |	0.4835        |	0.4838        |	0.4916        |
| 0   | 15      | 5          | 300         | 6       | 8       | 0.4963       |	0.4895        |	0.4857        |	0.484         |	0.4838        |	0.4919        |
| 0   | 15      | 5          | 300         | 7       | 8       | 0.4968       |	0.4903        |	0.4874        |	0.4847        |	0.4846        |	0.4929        |
| 0   | 15      | 5          | 400         | 5       | 8       | 0.4954       |	0.4896        |	0.4855        |	0.4829        |	0.4837        |	0.491         |
| 0   | 15      | 5          | 400         | 6       | 8       | 0.496        |	0.4895        |	0.4859        |	0.4843        |	0.4842        |	0.4921        |
| 0   | 15      | 5          | 400         | 7       | 8       | 0.4998       |	0.4917        |	0.488         |	0.486         |	0.4854        |	0.4937        |
| 1   | 15      | 5          | 200         | 5       | 8       | 0.506        |	0.4989        |	0.4953        |	0.4933        |	0.4931        |	0.5024        |
| 1   | 15      | 5          | 200         | 6       | 8       | 0.5082       |	0.501         |	0.4973        |	0.4962        |	0.4955        |	0.5045        |
| 1   | 15      | 5          | 200         | 7       | 8       | 0.5123       |	0.5045        |	0.5009        |	0.4986        |	0.4983        |	0.5065        |
| 1   | 15      | 5          | 300         | 5       | 8       | 0.5125       |	0.5047        |	0.5005        |	0.4978        |	0.4975        |	0.5055        |
| 1   | 15      | 5          | 300         | 6       | 8       | 0.5154       |	0.5066        |	0.5026        |	0.50          |	0.4999        |	0.5074        |
| 1   | 15      | 5          | 300         | 7       | 8       | 0.5167       |	0.509         |	0.5046        |	0.5026        |	0.5017        |	0.5094        |
| 1   | 15      | 5          | 400         | 5       | 8       | 0.5124       |	0.5029        |	0.5001        |	0.4986        |	0.4981        |	0.5057        |
| 1   | 15      | 5          | 400         | 6       | 8       | 0.5137       |	0.5065        |	0.5027        |	0.5007        |	0.5001        |	0.5084        |
| 1   | 15      | 5          | 400         | 7       | 8       | 0.5162       |	0.5082        |	0.5042        |	0.5031        |	0.5021        |	0.5105        |

