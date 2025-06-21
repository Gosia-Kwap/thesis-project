"""
Script for generating perturbed outputs from language models
"""

import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from src.utils.log_functions import log_message
from src.model_handlers.perturbator import PerturbationGenerator as Perturbator
from prompts.CoT import trigger_phrases, prompt_dict
from src.utils.Enums import MODEL_MAP, LEVEL




class ModelPipeline:
    def __init__(self, args):
        self.args = args
        self._validate_arguments()

        torch.cuda.empty_cache()

        self.model = MODEL_MAP[self.args.model]
        self.data = self._load_data()
        self.prompt = self._find_prompt()

    def _validate_arguments(self):
        """Validate input arguments"""
        if self.args.model not in MODEL_MAP:
            raise ValueError(
                f"Invalid model: {self.args.model}. "
                f"Available options: {list(MODEL_MAP.keys())}"
            )

        if self.args.start >= self.args.end:
            raise ValueError("Start index must be less than end index")

    def _find_prompt(self):
        return prompt_dict[self.args.task]

    def _load_data(self) -> pd.DataFrame:
        """Load and prepare dataset"""
        data_path = Path("data") / f"{self.args.task}_rephrased.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = pd.read_json(data_path)
        df = pd.DataFrame({
            'text': data['Body'] if 'Body' in data else None,
            'question' : data['Question'],
            'rephrased' : data['Rephrased'] if 'Rephrased' in data else None,
            'label': data['Answer'],
            'answers' : data['Answers'] if 'Answers' in data else None,
        })
        log_message(f"Data loaded from: {data_path}, data size: {len(df.iloc[self.args.start:self.args.end])}", "INFO")

        return df.iloc[self.args.start:self.args.end]

    def run(self):
        """Execute the full pipeline"""
        log_message(f"Starting processing for rows {self.args.start} to {self.args.end}")

        perturbator = Perturbator(
            self.model,
            self.prompt,
            trigger_phrases,
            self.args.quantisation
        )

        results = []
        batch_size = 4

        for batch_start in tqdm(range(0, len(self.data), batch_size),
                                desc="Processing batches"):
            batch = self.data.iloc[batch_start:batch_start + batch_size]

            # Pre-allocate GPU memory
            torch.cuda.empty_cache()

            for _, row in batch.iterrows():
                try:
                    generated_answers = perturbator.generate_for_question(
                        text=row["text"],
                        question=row["question"],
                        answers=row['answers'],
                        num_samples=3,
                        rephrased_questions=row["rephrased"] if "rephrased" in row else None,
                    )
                    results.append({
                        "input": row["text"],
                        "generated_answers": generated_answers,
                        "expected_output": row["label"]
                    })
                except RuntimeError as e:
                    log_message(f"Error processing row {row.name}: {str(e)}")
                    # Reduce batch size if OOM occurs
                    batch_size = max(1, batch_size // 2)
                    break

        self._save_results(results)

    def _save_results(self, results: List[Dict]):
        """Save results to disk"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        results_df = pd.DataFrame(results)
        quantisation_suffix = f"_{self.args.quantisation}" if self.args.quantisation else ""
        base_name = f"{self.args.task}_perturbed_outputs_{self.args.model}_{self.args.start}_{self.args.end}{quantisation_suffix}"
        results_df.to_json(output_dir / f"{base_name}.json", orient="records", indent=2)


        log_message(f"Results saved to {output_dir}/{base_name}.*")

