import os

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
model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    token=token)

# split for questions and answers
df = pd.DataFrame(data['Body'] + ', ' + data['Question'], columns=['text'])
df['label'] = data['Answer']

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
        "expected_output": row["output"]
    })

results_df = pd.DataFrame(results)
results_df.to_json(f"results/SVAMP_perturbed_outputs_{model_name}.json", orient="records", indent=2)
results_df.to_csv("results/SVAMP_Gemma_27b_perturbation_results.csv", index=False)
print("Results saved to results/SVAMP_Gemma27_perturbation_results.csv")