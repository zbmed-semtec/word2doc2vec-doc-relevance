import argparse
import json

def prepare_from_npy(filepath_in: str, remove_stop_words: bool):
    '''
    Retrieves data from RELISH and TREC npy files, separating pmid and the document consisting of title and abstract..

    Parameters
    ----------
    filepath_in: str
        The filepath of the RELISH or TREC input npy file.
    filepath_in: bool
        Whether to remove stopwords (True) or not (False).
    Returns
    ----------
    list of str
        All pubmed ids associated to the paper.
    list of list of str
        All tokenized words within the title + abstract.
    '''
    import numpy as np
    doc = np.load(filepath_in, allow_pickle=True)
    pmids = []
    article_docs = []
    if remove_stop_words:
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        for line in doc:
            pmids.append(int(doc[line][0]))
            article_docs.append(np.ndarray.tolist(doc[line][1]))
            article_docs[line].extend(np.ndarray.tolist(doc[line][2]))
            article_docs[line] = [w for w in article_docs[line] if not w in stop_words]
    for line in range(len(doc)):
        pmids.append(int(doc[line][0]))
        article_docs.append(np.ndarray.tolist(doc[line][1]))
        article_docs[line].extend(np.ndarray.tolist(doc[line][2]))
    return (pmids, article_docs)

def generate_Word2Vec_model(article_doc: list, pmids: list, params: list, filepath_out: str, use_pretrained: bool):
    '''
    Generates a word2vec model from all RELISH or TREC sentences using gensim and saves it as a .model file.

    Parameters
    ----------
    article_doc: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    pmids: list of str
        A list of all appearing pubmed ids in the input dataset.
    params: dict
        A dictionary of the hyperparameters for the model.
    filepath_out: str
        The filepath for the resulting word2vec model file.
    use_pretrained: bool
        Whether to use a pretrained Word2Vec model.
    '''
    from gensim.models import Word2Vec
    sentence_list = []
    for index in range(len(pmids)):
        sentence_list.append(article_doc[index])
    params['sentences'] = sentence_list
    wv_model = None
    if use_pretrained:
        print("Pretraining")
    else:
        wv_model = Word2Vec(**params)
    wv_model.save(filepath_out)

def generate_document_embeddings(pmids: str, article_doc: list, directory_out: str, param_iteration: int, gensim_model_path: str = ""):
    '''
    Generates document embeddings from a titles and abstracts in a given paper using word2vec and calculating the cenroids of all given word embeddings.
    If no gensim model is given, the 'glove-wiki-gigaword-200' gensim model is used.
    
    Parameters
    ----------
    pmids: list of str
        The list of all pmids which are processed.
    article_doc: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    directory_out: str
        The filepath of the output directory of all .npy embeddings.
    param_iteration: int
        Iteration through paramter list.
    gensim_model_path: str (optional)
        The filepath of the custom gensimModel.
    '''
    import gensim.downloader as api
    import gensim.models as model
    import time
    import os

    st = time.time()

    word_vectors = None
    has_custom_model = gensim_model_path != ""
    if has_custom_model:
        word_vectors = model.Word2Vec.load(gensim_model_path)
    else:
        print('using pretrained model')
        word_vectors = api.load('word2vec-google-news-300')
    missing_words = 0
    iteration = 0
    document_embeddings = []
    for iteration in range(len(pmids)):
        # Retrieve word embeddings.
        embedding_list = []
        if(has_custom_model):
            for word in article_doc[iteration]:
                try:
                    embedding_list.append(word_vectors.wv[word])
                except:
                    missing_words += 1
        else:
            for word in article_doc[iteration]:
                try:
                    embedding_list.append(word_vectors[word])
                except:
                    missing_words += 1

        # Generate document embeddings from word embeddings using word-vector centroids.
        if len(embedding_list) == 0:
            # This can be caused by a high min-count parameter or missing vocabulary when using a pretrained model
            document_embeddings.append([])
            continue
        document = [0.0] * word_vectors.vector_size

        for dim in range(word_vectors.vector_size):
            for word_embeddings in embedding_list:
                document[dim] += word_embeddings[dim]
            document[dim] = document[dim] / len(embedding_list)
        document_embeddings.append(document)

    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    
    import pandas as pd
    df = pd.DataFrame(list(zip((pmids), document_embeddings)), columns =['pmids', 'embeddings'])
    df = df.sort_values('pmids')
    os.makedirs(f"{directory_out}/{param_iteration}", exist_ok=True)
    df.to_pickle(f'{directory_out}/{param_iteration}/embeddings.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
        help="Path to input RELISH tokenized .npy file")
    parser.add_argument("-o", "--output", type=str,
        help="Path to save embeddings pickle file")       
    parser.add_argument("-pj", "--params_json", type=str,
        help="File location of word2vec parameter list.")
    parser.add_argument("-up", "--use_pretrained", type=int,
        help="Whether to use a pretrained model or not")
    parser.add_argument("-s", "--rm_stopwords", type=int,
		help="Whether to remove stopwords or not") 
    args = parser.parse_args()

    params = []
    with open(args.params_json, "r") as openfile:
        params = json.load(openfile)
    
    model_output_File = ""
    if not args.use_pretrained:
        model_output_File = "./data/word2vec_model"

    for iteration in range(len(params)):
        pmids, article_doc = prepare_from_npy(args.input, args.rm_stopwords)
        generate_Word2Vec_model(article_doc, pmids, params[iteration], model_output_File, args.use_pretrained)
        generate_document_embeddings(pmids, article_doc, args.output, iteration, model_output_File)