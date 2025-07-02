import os
from typing import List, Dict, Optional
import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from src.utils.log_functions import log_message
from src.utils.Enums import LEVEL, MODEL_MAP
from dotenv import load_dotenv
from src.model_handlers.ModelWrappers import TransformersModelWrapper, LlamaCppModelWrapper

load_dotenv()


class PerturbationGenerator:
    def __init__(self, model_name_or_path, generic_prompt: str, trigger_phrases: List[str],
                 level: LEVEL = LEVEL.INFO, quantisation=None, backend: str = "transformers"):
        self.level = level
        self.prompt = generic_prompt
        self.trigger_phrases = trigger_phrases

        if backend == "transformers":
            self.model_wrapper = TransformersModelWrapper(model_name_or_path, quantisation)
        elif backend == "llama_cpp":
            self.model_wrapper = LlamaCppModelWrapper(model_name_or_path)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.temperatures = [0.8, 0.9, 1.0]

    def generate_for_question(self, text: str, question: str, num_samples: int = 3,
                              rephrased_questions: list = None, answers=None) -> Dict[str, List[str]]:
        results = {}
        full_question = text + ". " + question
        if answers:
            full_question += f" {answers} "

        # Temperature perturbations
        for temp in self.temperatures:
            key = f"temp_{temp:.2f}"
            results[key] = self._generate_samples(
                full_question,
                num_samples,
                temperature=temp
            )

        # Trigger phrase perturbations
        for trigger in self.trigger_phrases:
            key = f"trigger_{trigger[:10]}"
            question_with_trigger = full_question + " " + trigger
            results[key] = self._generate_samples(
                question_with_trigger,
                num_samples
            )

        # Rephrased questions
        if rephrased_questions:
            for idx, rephrased in enumerate(rephrased_questions):
                q = text + ' ' + rephrased
                if answers:
                    q = q + ' ' + answers
                key = f"rephrased_{idx}"
                results[key] = self._generate_samples(q, 2)

        # Original
        results["original_answer"] = self._generate_samples(full_question, 1)

        return results

    def _generate_samples(self, question: str, num_samples: int, temperature: Optional[float] = None) -> List[str]:
        full_prompt = f"{self.prompt} {question} Answer: "
        return self.model_wrapper.generate(full_prompt, num_samples, temperature)
