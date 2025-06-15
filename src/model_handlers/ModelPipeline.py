"""
Script for generating perturbed outputs from language models
"""

import os
from pathlib import Path

# # Set these BEFORE importing transformers
# scratch_dir = Path(f"/scratch/{os.environ['USER']}/huggingface")
# tmp_dir = Path(f"/scratch/{os.environ['USER']}/tmp")
# scratch_dir.mkdir(parents=True, exist_ok=True)
# tmp_dir.mkdir(parents=True, exist_ok=True)
#
# os.environ.update({
#     "HF_HOME": str(scratch_dir),
#     "HF_HUB_CACHE": str(scratch_dir),
#     "TMPDIR": str(tmp_dir),
#     "TOKENIZERS_PARALLELISM": "false",
#     "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
# })

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
        self.device = self._get_device()

        torch.cuda.empty_cache()

        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.data = self._load_data()
        self.prompt = self._find_prompt()

    def _setup_environment(self):
        """Configure environment variables and directories"""
        self.scratch_dir = Path(f"/scratch/{os.environ['USER']}/huggingface")
        self.tmp_dir = Path(f"/scratch/{os.environ['USER']}/tmp")

        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def _validate_arguments(self):
        """Validate input arguments"""
        if self.args.model not in MODEL_MAP:
            raise ValueError(
                f"Invalid model: {self.args.model}. "
                f"Available options: {list(MODEL_MAP.keys())}"
            )

        if self.args.start >= self.args.end:
            raise ValueError("Start index must be less than end index")

    def _get_device(self) -> str:
        """Determine available device"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate configuration"""
        model_name = MODEL_MAP[self.args.model]

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.getenv("HUGGING_FACE_TOKEN"),
            cache_dir=os.getenv("HF_HOME"),
            use_fast=True,
            padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Model configuration
        model_kwargs = {
            "device_map": "auto",
            "token": os.getenv("HUGGING_FACE_TOKEN"),
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,
            "cache_dir" : os.getenv("HF_HOME")
        }

        if self.args.quantisation:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["lm_head"],  # Critical for stability
                # llm_int8_enable_fp32_cpu_offload=True
            )

        log_message("Loading model...", LEVEL.INFO)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Warmup the model
        if torch.cuda.is_available():
            with torch.no_grad():
                _ = model.generate(**tokenizer("Warmup", return_tensors="pt").to(self.device),
                                   max_new_tokens=1)

        return model, tokenizer

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
            self.tokenizer,
            self.prompt,
            trigger_phrases
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

