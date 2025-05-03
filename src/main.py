# Load dataset
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model_handlers.perturbator import PerturbationGenerator as Perturbator
from prompts.CoT import generic_prompt, trigger_phrases


load_dotenv()
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")
# Get the absolute path to the project directory
project_dir = os.getcwd()
token = os.getenv("HUGGING_FACE_TOKEN")

# Construct the full path to the dataset
data_path = os.path.join(project_dir, "data", "SVAMP.json")

# Load the dataset
data = pd.read_json(data_path)

# Step 3: Load the Gemma 2B Model and Tokenizer
model_name = "google/gemma-2-27b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=token)

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