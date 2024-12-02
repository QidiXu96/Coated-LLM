import anthropic
from pathlib import Path
from src.reviewer import reviewer, process_reviewer_multiple_runs

client = anthropic.Anthropic(
    api_key = "YOUR_API_KEY"
)  # Replace with your actual key

# Define the input CoT JSON file and results directory
CoT_json_file = "demo_data/chain_of_thoughts.json"  # JSON file with chains of thought
results_path = "demo_data/feedback_results"  # Directory to save the feedback results

# ToT Reviewer
process_reviewer_multiple_runs(CoT_json_file, client, results_path=results_path)

