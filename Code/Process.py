import sys
import logging

def prepareFromNPY(filepathIn=None):
        '''
        Retrieves data from RELISH and TREC npy files, separating each column into their own respective list.

        Parameters
        ----------
        filepathIn: str
                The filepath of the RELISH or TREC input npy file.

        Returns
        -------
        list of str
                All pubmed ids associated to the paper.
        list of str
                All words within the title.
        list of str
                All words within the abstract.
        '''
        if not isinstance(filepathIn, str):
                logging.alert("Wrong parameter type for prepareFromTSV.")
                sys.exit("filepathIn needs to be of type string")
        else:
                import numpy as np
                doc = np.load(filepathIn, allow_pickle=True)
                pmids = []
                titles = []
                abstracts = []
                for line in doc:
                        pmids.append(np.ndarray.tolist((int)(line[0])))
                        titles.append(np.ndarray.tolist(line[1]))
                        abstracts.append(np.ndarray.tolist(line[2]))
                return (pmids, titles, abstracts)

def generateWord2VecModel(titles, abstracts, filepathOut):
        '''
        Generates a word2vec model from all RELISH or TREC sentences using gensim and saves it as a .model file.

        Parameters
        ----------
        titles: list of str
                A two dimensional list of all titles and its words.
        abstracts: list of str
                A two dimensional list of all abstracts and its words.
        filepathOut: str
                The filepath for the resulting word2vec model file.
        '''
        if not isinstance(titles, list):
                logging.alert("Wrong parameter type for generateWord2VecModel.")
                sys.exit("titles needs to be of type list")
        elif not isinstance(abstracts, list):
                logging.alert("Wrong parameter type for generateWord2VecModel.")
                sys.exit("abstracts needs to be of type list")
        elif not isinstance(filepathOut, str):
                logging.alert("Wrong parameter type for generateWord2VecModel.")
                sys.exit("filepathOut needs to be of type string")
        else:
                from gensim.models import Word2Vec
                sentenceList = []
                for sentence in titles:
                        sentenceList.append(sentence)
                for sentence in abstracts:
                        sentenceList.append(sentence)
                model = Word2Vec(sentences=sentenceList, vector_size=400, epochs=5, window=7, min_count=1, workers=8)
                model.save(filepathOut)

def generateDocumentEmbeddings(pmids=None, titles=None, abstracts=None, directoryOut=None, gensimModelPath=None, saveAs="numpy"):
        '''
        Generates document embeddings from a titles and abstracts in a given paper using word2vec and calculating the cenroids of all given word embeddings.
        If no gensim model is given, the 'glove-wiki-gigaword-200' gensim model is used.
        
        Parameters
        ----------
        pmids: list of str
                The list of all pmids which are processed.
        titles: list of str
                The list of all titles which are processed.
        abstracts: list of str
                The list of all abstracts which are processed.
        directoryOut: str
                The filepath of the output directory of all .npy embeddings.
        gensimModelPath: str (optional)
                The filepath of the custom gensimModel.
        saveAs: str
                Can be saved as either 'numpy' or 'pandas'.
        '''
        if not isinstance(pmids, list):
                logging.alert("Wrong parameter type for generateDocumentEmbeddings.")
                sys.exit("pmids needs to be of type list")
        elif not isinstance(titles, list):
                logging.alert("Wrong parameter type for generateDocumentEmbeddings.")
                sys.exit("titles needs to be of type list")
        elif not isinstance(abstracts, list):
                logging.alert("Wrong parameter type for generateDocumentEmbeddings.")
                sys.exit("abstracts needs to be of type list")
        elif not isinstance(directoryOut, str):
                logging.alert("Wrong parameter type for generateDocumentEmbeddings.")
                sys.exit("directoryOut needs to be of type string")
        elif not isinstance(saveAs, str) or not (saveAs == "numpy" or saveAs == "pandas"):
                logging.alert("Wrong parameter type or value for saveAs.")
                sys.exit("saveAs needs to be a string of either 'numpy' or 'pandas'")
        else:
                import gensim.downloader as api
                import gensim.models as model
                word_vectors = None
                hasCustomModel = isinstance(gensimModelPath, str)
                if hasCustomModel:
                        word_vectors = model.Word2Vec.load(gensimModelPath)
                else:
                        word_vectors = api.load('glove-wiki-gigaword-200')
                missingWords = 0
                iteration = 0
                documentEmbeddings = []
                while (iteration < len(pmids)):
                        #Retrieve word embeddings.
                        embeddingList = []
                        first = True
                        for word in titles[iteration]:
                                try:
                                        if(hasCustomModel):
                                                embeddingList.append(word_vectors.wv[word])
                                        else:
                                                embeddingList.append(word_vectors[word])
                                except:
                                        missingWords += 1
                        for word in abstracts[iteration]:
                                try:
                                        if(hasCustomModel):
                                                embeddingList.append(word_vectors.wv[word])
                                        else:
                                                embeddingList.append(word_vectors[word])
                                except:
                                        missingWords += 1
                        #Generate document embeddings from word embeddings.
                        first = True
                        document = []
                        for embedding in embeddingList:
                                if first:
                                        for dimension in embedding:
                                                document.append(0.0)
                                        first = False
                                docDimension = 0
                                for dimension in embedding:
                                        document[docDimension] += dimension
                                        docDimension += 1
                        docDimension = 0
                        for dimension in document:
                                document[docDimension] = (dimension / len(embeddingList))
                                docDimension += 1

                        documentEmbeddings.append(document)
                        iteration += 1
                iteration = 0
                if(saveAs == 'pandas'):
                        import pandas as pd
                        df = pd.DataFrame(list(zip((pmids), documentEmbeddings)), columns =['pmids', 'embeddings'])
                        df = df.sort_values('pmids')
                        df.to_pickle(f'{directoryOut}/embeddings.pkl')
                elif(saveAs == 'numpy'):
                        import numpy as np
                        while(iteration < len(documentEmbeddings)):
                                np.save(f'{directoryOut}/{pmids[iteration]}', documentEmbeddings[iteration])
                                iteration += 1

def addCosineSimilarity(EvaluationFile, EmbeddingsDirectory):
        '''
        Adds cosine similarity to the evaluation matrix from the .npy formatted embeddings.

        Parameters
        ----------
        EvaluationFile: str
                The evaluation matrix csv file.
        EmbeddingsDirectory: str
                Directory in which document embeddings of each pmid is stored.
        '''
        if not isinstance(EvaluationFile, str):
                logging.alert("Wrong parameter type for addCosineSimilarity.")
                sys.exit("EvaluationFile needs to be of type string")
        elif not isinstance(EmbeddingsDirectory, str):
                logging.alert("Wrong parameter type for addCosineSimilarity.")
                sys.exit("EmbeddingsDirectory needs to be of type string")
        else:
                import csv
                import numpy as np
                from scipy import spatial
                with open(EvaluationFile, newline='') as csvfile:
                        spamreader = csv.reader(csvfile, delimiter=',')
                        header = True
                        headerRow = []
                        allRows = []
                        for row in spamreader:
                                if(header):
                                        header = False
                                        headerRow = row
                                else:
                                        try:
                                                reference = np.load(f'{EmbeddingsDirectory}/{row[0]}.npy', allow_pickle=True)
                                                assessed = np.load(f'{EmbeddingsDirectory}/{row[1]}.npy', allow_pickle=True)
                                                row[3] = round((1 - spatial.distance.cosine(reference, assessed)), 2)
                                        except:
                                                row[3] = ""
                                        allRows.append(row)
                        with open(EvaluationFile, 'w', newline='') as csvfile:
                                writer = csv.writer(csvfile, delimiter=',',)
                                writer.writerow(headerRow)
                                for row in allRows:
                                        writer.writerow(row)