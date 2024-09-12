import os
import argparse
import pandas as pd
from tqdm import tqdm
from scipy import spatial


def get_cosine_similarity(input_relevance_matrix: str, embeddings: str, output_matrix_name: str) -> None:
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
    """
    embeddings_df = pd.read_pickle(embeddings)
    relevance_matrix_df = pd.read_csv(input_relevance_matrix, sep="\t")
    # Adds the empty 4th column to the file
    print('Read embeddings pickle file')
    relevance_matrix_df["Cosine Similarity"] = ""

    # Create a dictionary to store embeddings
    embeddings_dict = {pmid: embedding for pmid, embedding in zip(embeddings_df['pmids'], embeddings_df['embeddings'])}

    # Create a list of ref and assessed PMID pairs
    pmid_pairs = list(zip(relevance_matrix_df["PMID1"], relevance_matrix_df["PMID2"]))

    cosine_similarities = []
    
    for ref_pmid, assessed_pmid in tqdm(pmid_pairs, total=len(pmid_pairs), desc="Calculating Cosine Similarities"):
        try:
            ref_pmid_vector = embeddings_dict[ref_pmid]
            assessed_pmid_vector = embeddings_dict[assessed_pmid]
            if ref_pmid_vector and assessed_pmid_vector:
                cosine_similarity = round(1 - spatial.distance.cosine(ref_pmid_vector, assessed_pmid_vector), 4)
        except:
            cosine_similarity = ""
        cosine_similarities.append(cosine_similarity)

    # Make changes in the original dataframe
    relevance_matrix_df['Cosine Similarity'] = cosine_similarities
    print('Added cosine scores')
    # Saves the cosine matrix 
    output_directory = os.path.dirname(output_matrix_name)
    if output_directory: 
        os.makedirs(output_directory, exist_ok=True)
        
    relevance_matrix_df.to_csv(output_matrix_name, index=False, sep="\t")
    print('Saved matrix')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="File path for the RELISH relevance matrix")
    parser.add_argument("-e", "--embeddings", type=str, help="File path for the embeddings in pickle format")
    parser.add_argument("-o", "--output", type=str, help="Output file path for generated 4 column cosine similarity matrix")
    args = parser.parse_args()
    get_cosine_similarity(args.input, args.embeddings, args.output)