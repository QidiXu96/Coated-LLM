import anthropic
from pathlib import Path
from moderator import moderator, process_final_multiple_runs

client = anthropic.Anthropic(
    api_key = "YOUR_KEY"
)  # Replace with your actual key

CoT_json_file = "chain_of_thoughts.json"  
feedback_path = "feedback_results/feedbacks_1.json"
results_path = "final_results" 

process_final_multiple_runs(CoT_json_file, feedback_path, client, runs=3, results_path=results_path)

