sudo pip install gdown
cd data

mkdir -p Input/Tokens
mkdir -p Input/Ground_truth

cd Input/Tokens
gdown https://drive.google.com/uc?id=1o_WoMuc-camPqPGHAMBA7Di2T9O8UYL0 -O relish.npy

cd ../Ground_truth
gdown https://drive.google.com/uc?id=1HYnBsx5xr-mqWMv8As3W7AzUlcx-ml22 -O relevance_matrix.tsv
