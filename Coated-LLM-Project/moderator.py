import anthropic
import json
from pathlib import Path
import time
import os

def moderator(client, question, response, feedback, model_name="claude-3-opus-20240229"):
    try:
        # Create the prompt with the provided question, response, and feedback
        prompt = "Previous response: " + response + "\n\n" + "Feedback: " + feedback + "\n\n" + "Based on the previous response and feedback, " + question + "Take a breath and work on this problem step by step. And conclude using the format 'Effective in theory: <Positive or Non-positive>.'"

        # Call the model to generate the final reasoning
        message = client.messages.create(
            model=model_name,
            max_tokens=1500,
            temperature=0.7,
            system="You are an expert in therapy development for Alzheimer's disease and you are trying to decide if the combination of two drugs is effective or not to treat or slow the progression of Alzheimer's disease in theory. As a proficient neurobiologist, use your own knowledge and search for external information if necessary. Also, it is rare that combination of two drugs become efficacious and synergistic in the real world.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the final reasoning from the model's response
        final_reasoning = message.content[0].text
        return final_reasoning
    except Exception as e:
        return str(e)

def process_final_multiple_runs(reasoning_file, feedback_file, client, batch_size=10, runs=5, results_path='.'):
    # Load chains of thoughts and feedback from the provided files
    with open(reasoning_file, 'r') as file:
        chains = json.load(file)
    with open(feedback_file, 'r') as file:
        fb = json.load(file)
        
    keys = list(chains.keys())
    total = len(keys)

    # Ensure the results path exists
    os.makedirs(results_path, exist_ok=True)

    for run in range(1, runs + 1):
        final_reasoning = {}
        for i in range(0, total, batch_size):
            batch_keys = keys[i:i + batch_size]
            for key in batch_keys:
                question = chains[key]['question']
                response = chains[key]['chain_of_thoughts']
                feedback = fb[key]
                final_reasoning[key] = moderator(client, question, response, feedback)
                time.sleep(2)  # Sleep to prevent hitting rate limits
        
        # Save the results of the current run to the specified path
        result_file = os.path.join(results_path, f'final_answer_with_feedback_{run}.json')
        with open(result_file, 'w') as f:
            json.dump(final_reasoning, f, indent=4)
        
        print(f"Run {run} completed and saved to {result_file}.")

    print("All runs processed and saved.")
