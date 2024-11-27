import h5py
import json
import pandas as pd
from pathlib import Path
from openai_utils import set_open_params, get_completion

def get_warmup_prompt(drugA, drugB, model, info_A, info_B):
    input_params = f"Drug A Name: {drugA}, Drug B Name: {drugB}, Model Name: {model}, Background Info: {info_A, info_B}"
    prompt = ("Background: " + info_A + " "+ info_B + 
              " Decide if the combination of " + drugA + " and " + drugB + 
              " is effective or not to treat " + model + " model in theory. "  +
              "Take a breath and work on this problem step by step. " +
              "And conculde using the format 'Effective in theory: <Positive or Non-positive>.'") 
    
    messages = [
        {
          "role": "system",
          "content": "You are an expert in therapy development for Alzheimer's disease and you are trying to decide if the combination of two drugs is effective or not to treat or slow the progression of Alzheimer's disease in theory.\
          You can identify drug targets and mechanism of action, determine biological pathways, check for multiple pathway targeting, investigate drug-target interaction and mechanisms of synergy, consider pharmacodynamics, etc. \
          Also, it is rare that combination of two drugs become efficacious and synergistic in real world. \
          As a proficient neurobiologist, use your own knowledge and search for external information if necessary."
        }, 
        {
            "role": "user",
            "content": prompt
        }
    ]
    return messages

def warmup(X_train, y_train, params, client, output_hdf5_path="embeddings.hdf5", pathway_dir="pathway", output_json_path="warm_up_CoT.json"):
    hdf5_file = h5py.File(output_hdf5_path, "w")
    text_data = {}

    for i in X_train.index:
        identifier = f"entry_{i}"
        file_path = Path(pathway_dir)
        
        # format pathway info
        drugA_name = X_train.loc[i, 'Drug A']
        drugB_name = X_train.loc[i, 'Drug B']
        
        drugA_pathway = file_path / f"{drugA_name}.csv"
        drugB_pathway = file_path / f"{drugB_name}.csv"
        
        A_formatted_pathway_terms = ""
        B_formatted_pathway_terms = ""
        
        if drugA_pathway.exists():
            A_pathway = pd.read_csv(drugA_pathway)
            A_formatted_pathway_terms = ', '.join(A_pathway['Pathway'].tolist())
        
        if drugB_pathway.exists():
            B_pathway = pd.read_csv(drugB_pathway)
            B_formatted_pathway_terms = ', '.join(B_pathway['Pathway'].tolist())
    
        info_A = f"{drugA_name} has several pathway information: {A_formatted_pathway_terms}" if A_formatted_pathway_terms else f""
        info_B = f"{drugB_name} has several pathway information: {B_formatted_pathway_terms}" if B_formatted_pathway_terms else f""
        
        # question & question embedding
        question = f"Decide if the combination of {X_train.loc[i, 'Drug A']} and {X_train.loc[i, 'Drug B']} is effective or not to treat {X_train.loc[i, 'Animal Model']} model in theory."
        embedding_response = client.embeddings.create(model="text-embedding-ada-002", input=question)
        question_embedding = embedding_response.data[0].embedding

        # chain of thoughts
        message = get_warmup_prompt(X_train.loc[i, 'Drug A'], X_train.loc[i, 'Drug B'], X_train.loc[i, 'Animal Model'], info_A, info_B)
        response = get_completion(params, message)
        chain_of_thoughts = response.choices[0].message.content

        # real answer
        real_answer = y_train.loc[i]

        # store the embedding with the identifier
        hdf5_file.create_dataset(identifier, data=question_embedding)

        # use the same identifier for the text data
        text_data[identifier] = {
            "question": question,
            "chain_of_thoughts": chain_of_thoughts,
            "real_answer": real_answer
        }

    hdf5_file.close()
    with open(output_json_path, "w") as json_file:
        json.dump(text_data, json_file, indent=4)

    return output_hdf5_path, output_json_path
