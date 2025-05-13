#!/usr/bin/env python3
"""
Script for generating perturbed outputs from language models
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from src.utils.log_functions import log_message
from src.model_handlers.perturbator import PerturbationGenerator as Perturbator
from prompts.CoT import generic_prompt, trigger_phrases
from src.utils.Enums import MODEL_MAP




class ModelPipeline:
    def __init__(self, args):
        self.args = args
        self._setup_environment()
        self._validate_arguments()
        self.device = self._get_device()
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.data = self._load_data()

    def _setup_environment(self):
        """Configure environment variables and directories"""
        self.scratch_dir = Path(f"/scratch/{os.environ['USER']}/huggingface")
        self.tmp_dir = Path(f"/scratch/{os.environ['USER']}/tmp")

        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        os.environ.update({
            "HF_HOME": str(self.scratch_dir),
            "HF_HUB_CACHE": str(self.scratch_dir),
            "TMPDIR": str(self.tmp_dir)
        })

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

        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.getenv("HUGGING_FACE_TOKEN")
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Model configuration
        if self.args.quantisation:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                load_in_8bit_fp32_cpu_offload=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                token=os.getenv("HUGGING_FACE_TOKEN")
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                token=os.getenv("HUGGING_FACE_TOKEN")
            )

        return model, tokenizer

    def _load_data(self) -> pd.DataFrame:
        """Load and prepare dataset"""
        data_path = Path("data") / f"{self.args.task}.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = pd.read_json(data_path)
        df = pd.DataFrame({
            'text': data['Body'] + ', ' + data['Question'],
            'label': data['Answer']
        })

        return df.iloc[self.args.start:self.args.end]

    def run(self):
        """Execute the full pipeline"""
        log_message(f"Starting processing for rows {self.args.start} to {self.args.end}")

        perturbator = Perturbator(
            self.model,
            self.tokenizer,
            generic_prompt,
            trigger_phrases
        )

        results = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            generated_answers = perturbator.generate_for_question(
                question=row["text"],
                num_samples=3,
            )
            results.append({
                "input": row["text"],
                "generated_answers": generated_answers,
                "expected_output": row["label"]
            })

        self._save_results(results)

    def _save_results(self, results: List[Dict]):
        """Save results to disk"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        base_name = f"{self.args.task}_perturbed_outputs_{self.args.model}_{self.args.start}_{self.args.end}"

        results_df = pd.DataFrame(results)
        results_df.to_json(output_dir / f"{base_name}.json", orient="records", indent=2)
        results_df.to_csv(output_dir / f"{base_name}.csv", index=False)

        log_message(f"Results saved to {output_dir}/{base_name}.*")

