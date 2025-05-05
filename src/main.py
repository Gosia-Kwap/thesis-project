import os
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, required=True, help="Start index of the dataset to process")
parser.add_argument("--end", type=int, required=True, help="End index (exclusive) of the dataset to process")
parser.add_argument("--model", type=str, required=True, help="Short model name key (e.g., gemma9b)")
args = parser.parse_args()

model_map = {
    "gemma9b": "google/gemma-2-9b-it",
    "gemma27b": "google/gemma-2-27b-it",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# Resolve model name
if args.model not in model_map:
    raise ValueError(f"Model name '{args.model}' not found in model map. Available keys: {list(model_map.keys())}")
model_name = model_map[args.model]

# Define paths
scratch_dir = f"/scratch/{os.environ['USER']}/huggingface"
tmp_dir = f"/scratch/{os.environ['USER']}/tmp"

# Create directories
os.makedirs(scratch_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

# Set Hugging Face cache and temp dir
os.environ["HF_HOME"] = scratch_dir
os.environ["HF_HUB_CACHE"] = scratch_dir  # also needed sometimes
os.environ["TMPDIR"] = tmp_dir

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model_handlers.perturbator import PerturbationGenerator as Perturbator
from prompts.CoT import generic_prompt, trigger_phrases

# Huggungface authorization
load_dotenv()
token = os.getenv("HUGGING_FACE_TOKEN")

# Dataset loading
project_dir = os.getcwd()
data_path = os.path.join(project_dir, "data", "SVAMP.json")
data = pd.read_json(data_path)

# Load model and its tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    token=token)

# split for questions and answers and cut for batching
df = pd.DataFrame(data['Body'] + ', ' + data['Question'], columns=['text'])
df['label'] = data['Answer']
df = df.iloc[args.start:args.end]

perturbator = Perturbator(model, tokenizer, generic_prompt, trigger_phrases)

results = []
for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    input_text = row["text"]
    # Generate answers using the Perturbator
    generated_answers = perturbator.generate_for_question(
        question=input_text,
        num_samples=3,
    )
    # Store results
    results.append({
        "input": input_text,
        "generated_answers": generated_answers,  # store all samples
        "expected_output": row["label"]
    })

os.makedirs("results", exist_ok=True)
output_json = f"results/SVAMP_perturbed_outputs_{args.model}_{args.start}_{args.end}.json"
output_csv = f"results/SVAMP_perturbed_outputs_{args.model}_{args.start}_{args.end}.csv"

results_df = pd.DataFrame(results)
results_df.to_json(output_json, orient="records", indent=2)
results_df.to_csv(output_csv, index=False)