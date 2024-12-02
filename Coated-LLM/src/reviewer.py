import anthropic
import json
from pathlib import Path
import time
import os

def reviewer(client, content_chain, model_name="claude-3-opus-20240229"):
    try:
        prompt = "Previous response: " + content_chain + "\n\n" + "Please evaluate the response. Explore the potential for drug interactions that could limit or enhance effectiveness."

        # Call the model to generate a review
        message = client.messages.create(
            model=model_name,
            max_tokens=1500,
            temperature=0.7,
            system="Imagine three different experts who are in therapy development for Alzheimer's disease, are tasked with critically reviewing the reasoning and conclusions regarding the effectiveness of a combination of two drugs on an Alzheimer's disease animal model from a theoretical perspective. \
            All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realizes they're wrong at any point, then they leave. \
            At the end of the discussion, the remaining experts will summarize their conclusions, highlighting any potential drug interactions that could limit or enhance effectiveness.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the feedback from the model's response
        feedback = message.content[0].text
        return feedback
    except Exception as e:
        return str(e)
    
def process_reviewer_multiple_runs(CoT_json_file, client, batch_size=10, runs=1, results_path='.'):
    with open(CoT_json_file, 'r') as file:
        entries = json.load(file)  # Selected chain_of_thoughts to send for review

    keys = list(entries.keys())
    total = len(keys)

    os.makedirs(results_path, exist_ok=True)

    for run in range(1, runs + 1):
        feedbacks = {}
        for i in range(0, total, batch_size):
            batch_keys = keys[i:i + batch_size]
            for key in batch_keys:
                feedbacks[key] = reviewer(client, entries[key]['chain_of_thoughts'])
                time.sleep(2)  # Sleep to prevent hitting rate limits

        # Save feedback to a JSON file in the results path
        result_file = os.path.join(results_path, f'feedbacks_{run}.json')
        with open(result_file, 'w') as f:
            json.dump(feedbacks, f, indent=4)

        print(f"Run {run} completed and saved to {result_file}.")

    print("All runs processed and saved.")