import argparse
import pandas as pd
import csv
from scipy import spatial
from multiprocessing import Pool, freeze_support
import os
import shutil
global_embeddings_df = None
try:
    print("Loading Pickle file")
    global_embeddings_df = pd.read_pickle("./data/embeddings.pkl")
except:
    print("Pickle file currently unknown, attempting to access Pickle file.")


def get_similarity(pair: list):
    """
    Calculates cosine similarity between two articles.
    Parameters
    ----------
    pair : list of str
        PMID pair of two articles.
    Returns
    ----------
    float
        Cosine similarity score.
    """
    return round(1 - spatial.distance.cosine(pair[0], pair[1]), 4)


def get_cosine_similarity(input_relevance_matrix: str, embeddings: str, direct_path: bool, output_matrix_name: str, iteration: int = -1) -> None:
    """
    Creates a 4 column matrix by appending cosine similarity scores for all existing pairs
    of PMIDs to the Relevance matrix.
    Parameters
    ----------
    input_relevance_matrix : str
        File path for relevance matrix (TREC/RELISH).
    embeddings : str
        File path for pickle file of pmids and embeddings.
    direct_path : bool
        Whether embeddings path is a direct path or the directory.
    output_matrix_name : str
        File path for the generated 4 column matrix.
    iteration: int (optional)
        If multiple matrices should be created, iteration is the suffix of the resulting matrix.
    """
    if direct_path:  # Embeddings path should be static to synchronize across processes
        if not os.path.isfile("./data/embeddings.pkl"):
            shutil.copy(embeddings, "./data/embeddings.pkl")
    else:
        shutil.copy(f"{embeddings}/{iteration}/embeddings.pkl",
                    "./data/embeddings.pkl")
    global_embeddings_df = pd.read_pickle("./data/embeddings.pkl")
    tokenpairs = []
    header = []
    rows = []
    with open(input_relevance_matrix, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        header = next(spamreader)  # Save and remove header
        for row in spamreader:
            try:
                tokenpairs.append([global_embeddings_df.loc[global_embeddings_df.pmids == int(row[0]),
                                                            "embeddings"].iloc[0], global_embeddings_df.loc[global_embeddings_df.pmids == int(row[1]), "embeddings"].iloc[0]])
                rows.append(row)
            except:
                print(f"PMID {row[0]} or PMID {row[1]} not found.")
                continue

    output_file = ""

    if direct_path:
        output_file = output_matrix_name
    else:
        output_file = f"{output_matrix_name}_{iteration}.tsv"

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)

        total_processed = 0
        with Pool() as p:
            iterator = p.imap(get_similarity, tokenpairs, 100)
            for similarity in iterator:
                row = rows[total_processed]
                row[3] = similarity
                writer.writerow(row)

                total_processed += 1
                if total_processed % 100 == 0 or total_processed == len(tokenpairs):
                    print(
                        f"Processed {total_processed}/{len(tokenpairs)} rows...")
        p.join()
        p.close()
    print('Saved matrix')


if __name__ == "__main__":
    __spec__ = None
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="File path for the TREC-repurposed/RELISH relevance matrix")
    parser.add_argument("-e", "--embeddings", type=str,
                        help="File path for the embeddings in pickle format")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file path for generated 4 column cosine similarity matrix")
    parser.add_argument("-c", "--doc_embeddings_count", type=int,
                        help="Number of docoument embeddings that have been created")
    args = parser.parse_args()
    if (not args.doc_embeddings_count):
        freeze_support()
        get_cosine_similarity(args.input, args.embeddings, True, args.output)
    else:
        for iteration in range(args.doc_embeddings_count):
            freeze_support()
            get_cosine_similarity(
                args.input, args.embeddings, False, args.output, iteration)
