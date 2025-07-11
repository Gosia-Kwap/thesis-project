import os
from typing import List, Dict, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

from src.utils.log_functions import log_message
from src.utils.Enums import LEVEL

from llama_cpp import Llama  # llama-cpp-python for GGUF models

load_dotenv()


class BaseModelWrapper:
    def generate(self, prompt: str, num_samples: int, temperature: Optional[float]) -> List[str]:
        raise NotImplementedError


class TransformersModelWrapper(BaseModelWrapper):
    def __init__(self, model_name, quantisation=None):
        self.model_name = model_name
        self.quantisation = quantisation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=os.getenv("HUGGING_FACE_TOKEN"),
            cache_dir=os.getenv("HF_HOME"),
            use_fast=True,
            padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model_kwargs = {
            "device_map": "auto",
            "token": os.getenv("HUGGING_FACE_TOKEN"),
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,
            "cache_dir": os.getenv("HF_HOME")
        }

        if self.quantisation:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["lm_head"]
            )

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        return model, tokenizer

    def generate(self, prompt: str, num_samples: int, temperature: Optional[float]) -> List[str]:
        inputs = self.tokenizer(prompt,
                                return_tensors="pt"
                                ).to(self.device)
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            temperature=temperature or 1.0,
            do_sample=True,
            num_return_sequences=num_samples,
            max_new_tokens=200,
            pad_token_id=self.tokenizer.eos_token_id
        )
        prompt_length = inputs.input_ids.shape[1]
        return [
            self.tokenizer.decode(seq[prompt_length:], skip_special_tokens=True).split("Answer:")[-1].strip()
            for seq in outputs
        ]


class LlamaCppModelWrapper(BaseModelWrapper):
    def __init__(self, model_path):
        try:
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=43,  # Offload ALL possible layers
                n_ctx=4096,
                main_gpu=0,  # Use primary GPU
                tensor_split=[1.0],  # Use 100% of GPU memory
                n_threads=8,  # Match your --cpus-per-task
                n_batch=512,  # Match your context window
                offload_kqv=True  # Special for Gemma models
            )
        except Exception as e:
            print(f"GPU offload failed ({str(e)}), falling back to CPU.")
            self.model = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=4096)

    def generate(self, prompt: str, num_samples: int, temperature: Optional[float]) -> List[str]:
        results = []
        for _ in range(num_samples):
            out = self.model(
                prompt,
                max_tokens=200,
                temperature=temperature or 1.0,
                stop=["<|endoftext|>"]
            )
            results.append(out["choices"][0]["text"].strip())
        return results


