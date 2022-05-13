import sys
import logging

def prepareFromTSV(filepathIn=None):
        '''
        Retrieves and formats data from the RELISH and TREC tsv files.

        To retrieve the correct word embeddings from word2vec modules, the spelling should be in lowercase letters only and all the stopwords need to be omitted.
        Once the input text has been processed, a list of all the words included in the title and abstract get returned.
        
        Input:  filepathIn      ->  string: The filepath of the RELISH or TREC input tsv file.

        Output: pmids           ->  list: A list of all pubmed ids (string) associated to the paper.
                titles          ->  list: A list of all words (string) within the title.
                abstrats        ->  list: A list of all words (string) within the abstract.
        '''
        if not isinstance(filepathIn, str):
                logging.alert("Wrong parameter type for prepareFromTSV.")
                sys.exit("filepathIn needs to be of type string")
        else:
                import csv
                import re
                from nltk import download
                from nltk.corpus import stopwords
                download('stopwords')
                stop_words = stopwords.words('english')
                pattern = '.*[a-zA-Z\d]{2,}.*' #Include all words which contain at least two letters or numbers.

                pmids = []
                titles = []
                abstracts = []
                with open(filepathIn) as input:
                        inputFile = csv.reader(input, delimiter="\t")
                        headerline = True
                        for line in inputFile:
                                if(headerline):
                                        headerline = False
                                else:
                                        title = line[1].lower().split()
                                        abstract = line[2].lower().split()
                                        iteration = 0
                                        while(iteration < len(title)):
                                                word = "".join([c for c in title[iteration] if re.match(pattern, c)])
                                                title[iteration] = word
                                                iteration += 1
                                        iteration = 0
                                        while(iteration < len(abstract)):
                                                word = "".join([c for c in abstract[iteration] if re.match(pattern, c)])
                                                abstract[iteration] = word
                                                iteration += 1
                                        titles.append([w for w in title if w not in stop_words])
                                        abstracts.append([w for w in abstract if w not in stop_words])
                                        pmids.append(line[0])
                return(pmids, titles, abstracts)

def prepareFromXML(directoryPath=None):
        '''
        Retrieves and formats data from the RELISH and TREC xml files.

        To retrieve the correct word embeddings from word2vec modules, the spelling should be in lowercase letters only and all the stopwords need to be omitted.
        Once the input text has been processed, a list of all the words included in the title and abstract get returned.
        
        Input:  directoryPath   ->  string: The directory path of the RELISH or TREC input xml directory.

        Output: pmids           ->  list: A list of all pubmed ids (string) associated to the paper.
                titles          ->  list: A list of all words (string) within the title.
                abstracts        ->  list: A list of all words (string) within the abstract.
        '''
        if not isinstance(directoryPath, str):
                logging.alert("Wrong parameter type for prepareFromXML.")
                sys.exit("directoryPath needs to be of type string")
        else:
                import os
                import xml.etree.ElementTree as et
                from nltk import download
                from nltk.corpus import stopwords
                download('stopwords')
                stop_words = stopwords.words('english')
                pattern = '.*[a-zA-Z\d].*' #Include all words which contain at least one letter or number.

                pmids = []
                titles = []
                abstracts = []
                for file in os.listdir(directoryPath):
                        if(file != '.DS_Store'):
                                xmlTree = et.parse(os.path.join(directoryPath, file))
                                root = xmlTree.getroot()
                                pmids.append(root[0][0].text)
                                titles.append(root[0][1][2].text)
                                abstracts.append(root[0][2][2].text)
                for title in titles:
                        while(iteration < len(title)):
                                word = "".join([c for c in title[iteration] if re.match(pattern, c)])
                                title[iteration] = word
                                iteration += 1
                        iteration = 0
                        titles.append([w for w in title if w not in stop_words])
                for abstract in abstracts:
                        while(iteration < len(abstract)):
                                word = "".join([c for c in abstract[iteration] if re.match(pattern, c)])
                                abstract[iteration] = word
                                iteration += 1
                        abstracts.append([w for w in abstract if w not in stop_words])
                return pmids, titles, abstracts

def generateWord2VecModel(titlesRELISH, titlesTREC, abstractsRELISH, abstractsTREC, filepathOut):
        '''
        Generates a word2vec model from all RELISH and TREC sentences using gensim and saves it as a .model file.
        
        Input:  titlesRELISH    ->  list: A two dimensional list of all RELISH titles and its words (string).
                titlesTREC      ->  list: A two dimensional list of all TREC titles and its words (string).
                abstractsRELISH ->  list: A two dimensional list of all RELISH abstracts and its words (string).
                abstractsTREC   ->  list: A two dimensional list of all TREC abstracts and its words (string).
                filepathOut     ->  string: The filepath for the resulting word2vec model file.
        '''
        if not isinstance(titlesRELISH, list):
                logging.alert("Wrong parameter type for generateWord2VecModel.")
                sys.exit("titlesRELISH needs to be of type list")
        elif not isinstance(titlesTREC, list):
                logging.alert("Wrong parameter type for generateWord2VecModel.")
                sys.exit("titlesTREC needs to be of type list")
        elif not isinstance(abstractsRELISH, list):
                logging.alert("Wrong parameter type for generateWord2VecModel.")
                sys.exit("abstractsRELISH needs to be of type list")
        elif not isinstance(abstractsTREC, list):
                logging.alert("Wrong parameter type for generateWord2VecModel.")
                sys.exit("abstractsTREC needs to be of type list")
        elif not isinstance(filepathOut, str):
                logging.alert("Wrong parameter type for generateWord2VecModel.")
                sys.exit("filepathOut needs to be of type string")
        else:
                from gensim.models import Word2Vec
                sentenceList = []
                for sentence in titlesRELISH:
                        sentenceList.append(sentence)
                for sentence in titlesTREC:
                        sentenceList.append(sentence)
                for sentence in abstractsRELISH:
                        sentenceList.append(sentence)
                for sentence in abstractsTREC:
                        sentenceList.append(sentence)
                model = Word2Vec(sentences=sentenceList, vector_size=200, epochs=5, window=5, min_count=1, workers=4)
                model.save(filepathOut)

def generateDocumentEmbeddings(pmids=None, titles=None, abstracts=None, directoryOut=None, gensimModelPath=None, distributionTitle=1, distributionAbstract=4):
        '''
        Generates document embeddings from a titles and abstracts in a given paper using word2vec and calculating the cenroids of all given word embeddings.
        The title and abstract are calculated individually and will get averaged out by a given distribution where the default setting is 1:4 for titles.
        The final document embedding consists of the following distirbution: finalDoc = (distributionTitle * embeddingTitle + distributionAbstract * embeddingAbstract) / (distributionTitle + distributionAbstract)

        The filepath for the gensim model is optional and intended for a domain specific model or one that was trained on the RELISH and TREC data set directly,
        which will be preffered over the 'glove-wiki-gigaword-200' gensim model, should an embedding not be present in the given list of embeddings, those words will be ignored.
        
        Input:  pmids                   ->  list: The list of all pmids (string) which are processed.
                titles                  ->  list: The list of all titles (string) which are processed.
                abstracts               ->  list: The list of all abstracts (string) which are processed.
                directoryOut            ->  string: The filepath of the output directory of all .npy embeddings.
                distributionTitle       ->  int: The distribution of title for the final document embedding.
                distributionAbstract    ->  int: The distribution of the abstract for the final document embedding.
                gensimModelPath         ->  string: The filepath of the custom gensimModel.
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
        elif not isinstance(distributionTitle, int):
                logging.alert("Wrong parameter type for generateDocumentEmbeddings.")
                sys.exit("distributionTitle needs to be of type int")
        elif not isinstance(distributionAbstract, int):
                logging.alert("Wrong parameter type for generateDocumentEmbeddings.")
                sys.exit("distributionAbstract needs to be of type int")
        else:
                import numpy as np
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
                        embeddingsTitle = []
                        embeddingsAbstract = []
                        for word in titles[iteration]:
                                try:
                                        if(hasCustomModel):
                                                embeddingsTitle.append(word_vectors.wv[word])
                                        else:
                                                embeddingsTitle.append(word_vectors[word])
                                except:
                                        missingWords += 1
                        for word in abstracts[iteration]:
                                try:
                                        if(hasCustomModel):
                                                embeddingsAbstract.append(word_vectors.wv[word])
                                        else:
                                                embeddingsAbstract.append(word_vectors[word])
                                except:
                                        missingWords += 1
                        #Generate document embeddings from word embeddings.
                        first = True
                        documentTitle = []
                        for embedding in embeddingsTitle:
                                if first:
                                        for dimension in embedding:
                                                documentTitle.append(0.0)
                                        first = False
                                docDimension = 0
                                for dimension in embedding:
                                        documentTitle[docDimension] += dimension
                                        docDimension += 1
                        docDimension = 0
                        for dimension in documentTitle:
                                documentTitle[docDimension] = (dimension / len(embeddingsTitle)) * distributionTitle
                                docDimension += 1

                        first = True
                        documentAbstract = []
                        for embedding in embeddingsAbstract:
                                if first:
                                        for dimension in embedding:
                                                documentAbstract.append(0.0)
                                        first = False
                                docDimension = 0
                                for dimension in embedding:
                                        documentAbstract[docDimension] += dimension
                                        docDimension += 1
                        docDimension = 0
                        for dimension in documentAbstract:
                                documentAbstract[docDimension] = (dimension / len(embeddingsAbstract)) * distributionAbstract
                                docDimension += 1

                        docDimension = 0
                        for dimension in documentTitle:
                                documentAbstract[docDimension] += dimension
                                documentAbstract[docDimension] = documentAbstract[docDimension] / (distributionAbstract + distributionTitle)
                                docDimension += 1
                        documentEmbeddings.append(documentAbstract)
                        iteration += 1
                iteration = 0
                while(iteration < len(documentEmbeddings)):
                        np.save(f'{directoryOut}/{pmids[iteration]}', documentEmbeddings[iteration])
                        iteration += 1