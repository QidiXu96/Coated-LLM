import anthropic
from pathlib import Path
from reviewer import reviewer, process_reviewer_multiple_runs

client = anthropic.Anthropic(
    api_key = "YOUR_KEY"
)  # Replace with your actual key

# Define the input CoT JSON file and results directory
CoT_json_file = "chain_of_thoughts.json"  # JSON file with chains of thought
results_path = "feedback_results"  # Directory to save the feedback results

# ToT Reviewer
process_reviewer_multiple_runs(CoT_json_file, client, results_path=results_path)

