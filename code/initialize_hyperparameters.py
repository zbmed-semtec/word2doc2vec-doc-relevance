import json
import argparse

def store_hyperparameters(params: list, target_file: str):
    """
    Reads a list of word2vec training parameters and saves them as a JSON file.

    Parameters
    ----------
    params : list
        List of word2vec parameter options.
    target_file : str
        Path to the JSON hyperparameters file.
    """
    with open(target_file, 'w') as params_json:
        json.dump(params, params_json, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--target_file", type=str,
                    help="Path of target JSON file")         
    args = parser.parse_args()
    params = [{'vector_size':200, 'epochs':15, 'window':5, 'min_count':5, 'workers':8, 'sg':0}]

    store_hyperparameters(params = params, target_file = args.target_file)