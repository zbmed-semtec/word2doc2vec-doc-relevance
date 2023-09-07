import argparse
import pandas as pd
from scipy import spatial


def get_cosine_similarity(input_relevance_matrix: str, embeddings: str, output_matrix_name: str, corpus: str) -> None:
    """
    Creates a 4 column matrix by appending cosine similarity scores for all existing pairs
    of PMIDs to the Relevance matrix.
    Parameters
    ----------
    input_relevance_matrix : str
        File path for relevance matrix (TREC/RELISH).
    embeddings : str
        File path for pickle file of pmids and embeddings.
    output_matrix_name : str
        File path for the generated 4 column matrix.
    corpus : str
        Name of the corpus (TREC or RELISH).
    """
    embeddings_df = pd.read_pickle(embeddings)
    if corpus == "RELISH":
        column_names = ["PMID1", "PMID2", "Relevance"]
    elif corpus == "TREC":
        column_names = ["PMID1", "PMID2", "Rel-d2d"]
    relevance_matrix_df = pd.read_csv(input_relevance_matrix, names=column_names, sep="\t")
    # Adds the empty 4th column to the file
    print('Read embeddings pickle file')
    relevance_matrix_df["Cosine Similarity"] = ""

    for index, row in relevance_matrix_df.iterrows():
        # print(type(row["PMID1"]), type(row["PMID2"]))
        ref_pmid = str(row["PMID1"])
        assessed_pmid = str(row["PMID2"])
        try:
            # Determine the cosine similarity of the ref and assessed pmids and add to the 4th column
            ref_pmid_vector = embeddings_df.loc[embeddings_df.pmids == ref_pmid, "embeddings"].iloc[0]
            assessed_pmid_vector = embeddings_df.loc[embeddings_df.pmids == assessed_pmid, "embeddings"].iloc[0]
            row["Cosine Similarity"] = round(1 - spatial.distance.cosine(ref_pmid_vector, assessed_pmid_vector), 4)
        except:
            # Leave the 4th column empty if the ref or assessed pmid not found in the dataset
            row["Cosine Similarity"] = ""

        # Make changes in the original dataframe
        relevance_matrix_df.at[index, 'Cosine Similarity'] = row['Cosine Similarity']
    print('Added cosine scores')
    relevance_matrix_df.to_csv(output_matrix_name, index=False, sep="\t")
    print('Saved matrix')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="File path for the TREC-repurposed/RELISH relevance matrix")
    parser.add_argument("-e", "--embeddings", type=str, help="File path for the embeddings in pickle format")
    parser.add_argument("-o", "--output", type=str, help="Output file path for generated 4 column cosine similarity matrix")
    parser.add_argument("-c", "--corpus", type=str, help="Name of the corpus (TREC or RELISH)")
    args = parser.parse_args()
    get_cosine_similarity(args.input, args.embeddings, args.output, args.corpus)