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