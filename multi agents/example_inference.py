import pandas as pd
import numpy as np
import h5py
from inference import inference
from openai_utils import set_open_params
import openai
from openai import OpenAI

embedding_hdf5_path = "demo_embeddings_final.hdf5"  
json_path = "demo_warm_up_CoT.json"     
pathway_dir = "pathway"        
test_data_path = "demo_test.xlsx" 
output_dir = "demo_test_results"            

X_test = pd.read_excel(test_data_path)

with h5py.File(embedding_hdf5_path, 'r') as hdf5_file:
    embeddings = [hdf5_file[name][:] for name in hdf5_file.keys()]
    embedding_train = np.stack(embeddings) 

# Set OpenAI API parameters
params = set_open_params()

# Initialize OpenAI client
openai.api_key = "YOUR_API_KEY"  # Replace with your OpenAI API key
client = OpenAI(api_key='YOUR_API_KEY')

# Run inference
def run_multiple_inferences(X_test, iterations, n_neighbors, embedding_train, params, output_dir):
    result_files = []
    for iteration in range(1, iterations + 1):
        result_file_path = inference(
            X_test=X_test,
            n=n_neighbors,
            params=params,
            embedding_train=embedding_train,
            client=openai,
            iteration=iteration,
            hdf5_path=embedding_hdf5_path,
            json_path=json_path,
            pathway_dir=pathway_dir,
            output_dir=output_dir
        )
        result_files.append(result_file_path)
        print(f"Iteration {iteration} results saved to: {result_file_path}")
    return result_files

# Run inference for multiple iterations
iterations = 3  # Number of iterations
n_neighbors = 2  # Number of neighbors for k-NN
result_files = run_multiple_inferences(X_test, iterations, n_neighbors, embedding_train, params, output_dir)
