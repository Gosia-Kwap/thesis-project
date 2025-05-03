# Load dataset
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model_handlers.perturbator import PerturbationGenerator as Perturbator
from prompts.CoT import generic_prompt, trigger_phrases

# Huggungface authorization
load_dotenv()
token = os.getenv("HUGGING_FACE_TOKEN")

# Move the directory for the models to the more storage module
scratch_dir = f"/scratch/{os.getenv('USER')}/huggingface"
os.makedirs(scratch_dir, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = scratch_dir

# Dataset loading
project_dir = os.getcwd()
data_path = os.path.join(project_dir, "data", "SVAMP.json")
data = pd.read_json(data_path)

# Load model and its tokenizer
model_name = "google/gemma-2-27b-it"
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
    generated_answers = perturbator._generate_samples(
        task="question-answering",
        input_text=input_text,
        num_samples=1,  # Generate one answer per input
        temperature=0.7  # Use default temperature
    )
    # Store results
    results.append({
        "input": input_text,
        "generated_answer": generated_answers[0],  # Take the first generated answer
        "expected_output": row["output"]
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results/SVAMP_Gemma_27b_perturbation_results.csv", index=False)
print("Results saved to results/SVAMP_Gemma27_perturbation_results.csv")