import sys
import logging

def prepareFromTSV(filepathIn):
        import csv
        from nltk import download
        from nltk.corpus import stopwords
        download('stopwords')
        stop_words = stopwords.words('english')
        '''
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
                pmids = []
                titles = []
                abstracts = []
                with open(filepathIn) as input:
                        inputFile = csv.reader(input, delimiter="\t")
                        for line in inputFile:
                                title = line[1].lower().split()
                                abstract = line[2].lower().split()
                                titles.append([w for w in title if w not in stop_words])
                                abstracts.append([w for w in abstract if w not in stop_words])
                                pmids.append(line[0])
                return(pmids, titles, abstracts)

def generateDocumentEmbeddings(pmids, titles, abstracts, directoryOut, distributionTitle = 1, distributionAbstract = 4):
        '''
        Generates document embeddings from a titles and abstracts in a given paper using word2vec and calculating the mean value of all given word embeddings.
        The title and abstract are calculated individually and will get averaged out by a given distribution where the default setting is 1:4 for titles.
        The final document embedding consists of the following distirbution: finalDoc = (distributionTitle * embeddingTitle + distributionAbstract * embeddingAbstract) / (distributionTitle + distributionAbstract)
        
        Input:  pmids                   ->  list: The list of all pmids (string) which are processed.
                titles                  ->  list: The list of all titles (string) which are processed.
                abstracts               ->  list: The list of all abstracts (string) which are processed.
                directoryOut            ->  string: The filepath of the output directory of all .npy embeddings.
                distributionTitle       ->  int: The distribution of title for the final document embedding.
                distributionAbstract    ->  int: The distribution of the abstract for the final document embedding.
        '''
        if not isinstance(pmids, list):
                logging.alert("Wrong parameter type for generateDocumentEmbeddings.")
                sys.exit("pmids needs to be of type string")
        elif not isinstance(titles, list):
                logging.alert("Wrong parameter type for generateDocumentEmbeddings.")
                sys.exit("titles needs to be of type string")
        elif not isinstance(abstracts, list):
                logging.alert("Wrong parameter type for generateDocumentEmbeddings.")
                sys.exit("abstracts needs to be of type string")
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
                from gensim.models import KeyedVectors
                missingWords = 0
                word_vectors = KeyedVectors.load_word2vec_format('pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin', binary=True)
                iteration = 0
                documentEmbeddings = []
                while (iteration < len(pmids)):
                        #Retrieve word embeddings.
                        embeddingsTitle = []
                        embeddingsAbstract = []
                        for word in titles[iteration]:
                                try:
                                        embeddingsTitle.append(word_vectors[word])
                                except:
                                        missingWords += 1
                        for word in abstracts[iteration]:
                                try:
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
                        for dimension in documentTitle:
                                dimension = (dimension / len(embeddingsTitle)) * distributionTitle
                        
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
                        for dimension in documentAbstract:
                                dimension = (dimension / len(embeddingsAbstract)) * distributionAbstract

                        docDimension = 0
                        for dimension in documentTitle:
                                documentAbstract[docDimension] += dimension
                                documentAbstract[docDimension] = documentAbstract[docDimension] / (distributionAbstract + distributionTitle)
                                docDimension += 1

                        iteration += 1