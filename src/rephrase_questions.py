import argparse
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from pathlib import Path

def batch_rephrase_questions(questions, tokenizer, model, device, num_return_sequences=3, batch_size=16):
    all_rephrased = []
    for i in tqdm(range(0, len(questions), batch_size), desc="Batch Rephrasing"):
        batch = questions[i:i+batch_size]
        inputs = [f"paraphrase the question. The sense must be the same but use synonyms: {q} </s>" for q in batch]

        encodings = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                max_length=64,
                num_beams=10,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                temperature=1,
                top_k=50,
                top_p=0.95,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Group decoded into batches of `num_return_sequences`
        grouped = [decoded[j:j+num_return_sequences] for j in range(0, len(decoded), num_return_sequences)]
        all_rephrased.extend(grouped)

    return all_rephrased

def main(input_path='questions.json', output_path='questions_rephrased.json', batch_size=32):
    # Load data
    with open(input_path, 'r') as f:
        data = json.load(f)

    questions = [item['Question'] for item in data]

    # Load model and tokenizer
    model_name = "ramsrigouthamg/t5_paraphraser"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Rephrase in batches
    all_rephrased = batch_rephrase_questions(questions, tokenizer, model, device, batch_size=batch_size)

    # Add to original data
    for item, rephrased in zip(data, all_rephrased):
        item["Rephrased"] = rephrased

    # Save output
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rephrase questions in a dataset.")
    parser.add_argument("--task", type=str, required=True, help="Name of the dataset file (without .json)")

    args = parser.parse_args()

    input_path = Path("data") / f"{args.task}.json"
    output_path = Path("data") / f"{args.task}_rephrased.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Data file not found: {input_path}")

    main(input_path=str(input_path), output_path=str(output_path))
