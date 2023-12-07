# This script generates a compiled TSV file from the individual generated Precision and nDCG files for all hyperparameter combinations.

import re
import pandas as pd
import argparse
from os import listdir
from os.path import isfile, join

def compile_results(input_path, output_path):
    # Get all files in the folder
    all_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    # Sort files in numerical order
    all_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    combined_list = []

    # Loop through the files
    for index, file in enumerate(all_files):
        file_path = input_path + file
        # Read each tsv file
        df = pd.read_csv(file_path, sep='\t')
        # Remove the index column in the Pandas dataframe
        df = df.iloc[: , 1:]
        # Get only the last row (containing the average values)
        df = df.tail(1)
        # Convert the last row to list and append to 'combined_list'
        flattened = [val for sublist in df.values.tolist() for val in sublist]
        combined_list.append(flattened)

        # For the final loop, create a dataframe using 'combined_list' and write to tsv file
        if index == (len(all_files) - 1):
            combined_df = pd.DataFrame(combined_list, columns=df.columns.values.tolist())
            
            combined_df = combined_df.drop('PMIDs', axis=1)
            # print('vjk')
            combined_df.to_csv(output_path, sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help="Path to the directory consisting of all the gain matrices..")
    parser.add_argument('-o', '--output', type=str, 
                        help="Output path along with the name of the file to save the generated compiled nDCG@N TSV file.")
    args = parser.parse_args()
    compile_results(args.input, args.output)

   

