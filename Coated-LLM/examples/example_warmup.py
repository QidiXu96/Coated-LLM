import pandas as pd
from src.warmup import warmup
from src.openai_utils import set_open_params, get_completion
import openai
from openai import OpenAI

# read the data
train_data_path = "demo_data/warm_up_demo_data.xlsx"
data = pd.read_excel(train_data_path)

X_train = data[['Drug A', 'Drug B', 'Animal Model']]
y_train = data['Efficacy'] 

params = set_open_params()

openai.api_key = "YOUR_API_KEY"  # Replace with your OpenAI API key
client = OpenAI(api_key='YOUR_API_KEY')

output_hdf5_path = "demo_data/demo_embeddings.hdf5"
pathway_dir = "demo_data/pathway"  
output_json_path = "demo_data/demo_warm_up_CoT.json"

# Run the warmup function
hdf5_path, json_path = warmup(
    X_train,
    y_train,
    params,
    client,
    output_hdf5_path=output_hdf5_path,
    pathway_dir=pathway_dir,
    output_json_path=output_json_path
)
