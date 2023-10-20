import argparse

def prepare_from_npy(filepath_in: str):
        '''
        Retrieves data from RELISH and TREC npy files, separating pmid and the document consisting of title and abstract..

        Parameters
        ----------
        filepath_in: str
                The filepath of the RELISH or TREC input npy file.
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
        article_doc = []
        for line in doc:
                pmids.append(np.ndarray.tolist(line[0]))
                article_doc.append(np.ndarray.tolist(line[1]))
                article_doc.extend(np.ndarray.tolist(line[2]))
        return (pmids, article_doc)

def generate_Word2Vec_model(article_doc: list, params: list, filepath_out: str, use_pretrained: bool):
        '''
        Generates a word2vec model from all RELISH or TREC sentences using gensim and saves it as a .model file.

        Parameters
        ----------
        article_doc: list of list of str
                A two dimensional list of all tokenized article documents (title + abstract).
        params: dict
                A dictionary of the hyperparameters for the model.
        filepath_out: str
                The filepath for the resulting word2vec model file.
        use_pretrained: bool
                Whether to use a pretrained Word2Vec model.
        '''
        from gensim.models import Word2Vec
        params['sentences'] = article_doc
        wv_model = None
        if use_pretrained:
                print("Pretraining")
        else:
                wv_model = Word2Vec(**params)
        wv_model.save(filepath_out)

def generate_document_embeddings(pmids: str, article_doc: list, directory_out: str, gensim_model_path: str = ""):
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
        gensim_model_path: str (optional)
                The filepath of the custom gensimModel.
        '''
        import gensim.downloader as api
        import gensim.models as model
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
                #Retrieve word embeddings.
                embedding_list = []
                first = True
                for word in article_doc[iteration]:
                        try:
                                if(has_custom_model):
                                        embedding_list.append(word_vectors.wv[word])
                                else:
                                        embedding_list.append(word_vectors[word])
                        except:
                                missing_words += 1
                #Generate document embeddings from word embeddings using word-vector centroids.
                first = True
                document = []
                for embedding in embedding_list:
                        if first:
                                for dimension in embedding:
                                        document.append(0.0)
                                first = False
                        doc_dimension = 0
                        for dimension in embedding:
                                document[doc_dimension] += dimension
                                doc_dimension += 1
                doc_dimension = 0
                for dimension in document:
                        document[doc_dimension] = (dimension / len(embedding_list))
                        doc_dimension += 1

                document_embeddings.append(document)
        import pandas as pd
        df = pd.DataFrame(list(zip((pmids), document_embeddings)), columns =['pmids', 'embeddings'])
        df = df.sort_values('pmids')
        df.to_pickle(f'{directory_out}/embeddings.pkl')

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input", type=str,
                help="Path to input RELISH tokenized .npy file")
        parser.add_argument("-o", "--output", type=str,
                help="Path to save embeddings pickle file")                 
        args = parser.parse_args()

        params = {'vector_size':200, 'epochs':5, 'window':5, 'min_count':2, 'workers':4}
        model_output_File = "./data/word2vec_model"

        pmids, article_doc, docs = prepare_from_npy(args.input)
        generate_Word2Vec_model(article_doc, params, model_output_File)
        generate_document_embeddings(pmids, article_doc, args.output, model_output_File)