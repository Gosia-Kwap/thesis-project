from typing import List, Dict, Optional
import torch
from transformers import AutoConfig


class PerturbationGenerator:
    """Handles efficient generation of perturbed samples"""

    def __init__(self, model, tokenizer, generic_prompt: str, trigger_phrases: List[str]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-cache all prompt components
        self.base_prompt_ids = self.tokenizer.encode(generic_prompt, return_tensors='pt').to(self.device)
        self.trigger_cache = {
            trigger: self.tokenizer.encode(trigger, return_tensors='pt').to(self.device)
            for trigger in trigger_phrases
        }
        self.prompt = generic_prompt
        self.trigger_phrases = trigger_phrases

        # Default temperatures for perturbation
        self.default_temp = getattr(model.config, "temperature", None)
        if self.default_temp == 1.0:
            self.temperatures = [
                0.8,
                1.0,
                # 1.2,
            ]
        else:
            self.temperatures = [
                self.default_temp,
                self.default_temp + (1 - self.default_temp) / 2,
                1.0
            ]

    def generate_for_question(self, question: str, num_samples: int = 3) -> Dict[str, List[str]]:
        """Generate all perturbations for a single question"""
        results = {}

        # Temperature perturbations
        for temp in self.temperatures:
            key = f"temp_{temp:.2f}"
            results[key] = self._generate_samples(
                question=question,
                num_samples=num_samples,
                temperature=temp
            )

        # Trigger phrase perturbations
        for trigger in self.trigger_cache:
            key = f"trigger_{trigger[:10]}"
            results[key] = self._generate_samples(
                question=question,
                num_samples=num_samples,
                trigger_phrase=trigger
            )

        return results

    def _generate_samples(self, question: str, num_samples: int,
                          temperature: Optional[float] = None,
                          trigger_phrase: Optional[str] = None) -> List[str]:
        """Generates clean outputs without any prompt text"""
        # 1. Prepare all input components

        trigger = trigger_phrase if trigger_phrase else next(iter(self.trigger_phrases))
        full_prompt = self.prompt + question + trigger + ' '  'Answer: \n'

        # 2. Tokenize entire prompt once
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        device = self.model.device  # Make sure this is correct
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            # Forward pass to check for NaNs/infs in logits
            forward_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            print(f"[DEBUG] Input prompt: {repr(full_prompt)}")
            print(f"[DEBUG] Input IDs: {input_ids}")
            print(f"[DEBUG] Attention Mask: {attention_mask}")
            print(f"[DEBUG] Logits: {forward_outputs.logits}")

            if hasattr(forward_outputs, "logits"):
                logits = forward_outputs.logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError("NaN or Inf detected in model logits before sampling.")

            # Now do actual generation
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature or self.default_temp,
                do_sample=True,
                num_return_sequences=num_samples,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True
            )

        # 4. Cut out the prompt and decode
        prompt_length = inputs.input_ids.shape[1]
        clean_outputs = []

        for seq in outputs.sequences:
            generated_tokens = seq[prompt_length:]

            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            text = text.split("Answer:")[-1].strip()  # Remove any new answer prefixes

            clean_outputs.append(text)

        return clean_outputs

