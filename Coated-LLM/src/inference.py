import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from src.openai_utils import set_open_params, get_completion
from openai.types import CreateEmbeddingResponse, Embedding
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

def getsimilarexamples(drugA, drugB, animalmodel, n, embedding_train, client, hdf5_path, json_path):
    test_question = f"Decide if the combination of {drugA} and {drugB} is effective or not to treat {animalmodel} model in theory."
    
    # Generate embedding for the test question
    embedding_response = client.embeddings.create(model="text-embedding-ada-002", input=test_question)
    test_question_embedding = embedding_response.data[0].embedding
    X_test_reshape = np.array(test_question_embedding).reshape(1, -1)

    # Fit the k-NN model on training embeddings
    knn_model = NearestNeighbors(n_neighbors=n, metric='cosine')
    knn_model.fit(embedding_train)
    
    distances, indices = knn_model.kneighbors(X_test_reshape)
    most_similar_indices = indices.flatten()  # Get the indices of the most similar entries

    # Fetch the most similar entries from the embedding HDF5 file
    most_similar_keys = []
    with h5py.File(hdf5_path, 'r') as file:
        for index, key in enumerate(file.keys()):
            if index in most_similar_indices:
                most_similar_keys.append(key)
                if len(most_similar_keys) == len(most_similar_indices):  # Stop when all keys are found
                    break

    # Fetch the most similar records' details (question and CoT) from the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

    most_similar_records_details = []
    for key in most_similar_keys:
        if key in data:
            record = data[key]
            question = record.get('question', 'No question available')
            chain_of_thoughts = record.get('chain_of_thoughts', 'No chain of thoughts available')
            most_similar_records_details.append({'question': question, 'chain_of_thoughts': chain_of_thoughts})

    return test_question, most_similar_records_details

def get_inference_prompt(most_similar_records_details, test_question, infoA, infoB, params):
    # create few-shot example
    example_template = """
    Question: {question}
    Reasons: {chain_of_thoughts}
    """

    prompt = " "

    for record in most_similar_records_details:
        prompt += example_template.format(question=record['question'], chain_of_thoughts=record['chain_of_thoughts'])
        
    prompt += '\n\n' + "Background: " + infoA + infoB + '\n\n'
    prompt += test_question
    prompt += "Take a breath and work on this problem step by step. And conclude using the format 'Effective in theory: <Positive or Non-positive>.'"

    messages = [
    {
        "role": "system",
        "content": "You are an expert in therapy development for Alzheimer's disease and you are trying to decide if the combination of two drugs is effective or not to treat or slow the progression of Alzheimer's disease in theory.\
        Also, it is rare that combination of two drugs become efficacious and synergistic. \
        As a proficient neurobiologist, use your own knowledge and search for external information if necessary."
    }, 
    {
        "role": "user",
        "content": prompt
    }]
    response = get_completion(params, messages)
    CoT = response.choices[0].message.content
        
    return CoT

def inference(X_test, n, params, embedding_train, client, iteration, hdf5_path, json_path, pathway_dir="pathway", output_dir="testing"):
    test_data = {}
    file_path = Path(pathway_dir)

    for i in X_test.index:
        identifier = f"entry_{i}"

        # Format pathway info
        drugA_name = X_test.loc[i, 'Drug A']
        drugB_name = X_test.loc[i, 'Drug B']

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

        info_A = f"{drugA_name} has several pathway information: {A_formatted_pathway_terms}" if A_formatted_pathway_terms else ""
        info_B = f"{drugB_name} has several pathway information: {B_formatted_pathway_terms}" if B_formatted_pathway_terms else ""

        test_question, most_similar_records_details = getsimilarexamples(drugA_name, drugB_name, X_test.loc[i, 'Animal Model'], n, embedding_train, client, hdf5_path, json_path)
                                                               
        CoT = get_inference_prompt(most_similar_records_details, test_question, info_A, info_B, params)
            
        test_data[identifier] = {
                "question": test_question,
                "chain_of_thoughts": CoT
        }

    result_file = Path(output_dir) / f"test_result_{iteration}.json"

    with open(result_file, "w") as json_file:
        json.dump(test_data, json_file, indent=4)

    return str(result_file)
