import anthropic
from pathlib import Path
from src.moderator import moderator, process_final_multiple_runs

client = anthropic.Anthropic(
    api_key = "YOUR_API_KEY"
)  # Replace with your actual key

CoT_json_file = "demo_data/chain_of_thoughts.json"  
feedback_path = "demo_data/feedback_results/feedbacks_1.json"
results_path = "demo_data/final_results" 

process_final_multiple_runs(CoT_json_file, feedback_path, client, runs=3, results_path=results_path)

