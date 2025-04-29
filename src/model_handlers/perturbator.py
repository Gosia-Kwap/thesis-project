from typing import List, Dict, Optional
import torch

class PerturbationGenerator:
    """Handles efficient generation of perturbed samples"""

    def __init__(self, model, tokenizer, generic_prompt: str, trigger_phrases: List[str], default_temp=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Pre-cache all prompt components
        self.base_prompt_ids = self.tokenizer.encode(generic_prompt, return_tensors='pt').to(self.device)
        self.trigger_cache = {
            trigger: self.tokenizer.encode(trigger, return_tensors='pt').to(self.device)
            for trigger in trigger_phrases
        }
        self.prompt = generic_prompt
        self.trigger_phrases = trigger_phrases

        # Default temperatures for perturbation
        self.default_temp = default_temp
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

        # 2. Tokenize entire prompt once (more accurate for length calculation)
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # 3. Generate with attention mask
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                temperature=temperature or self.default_temp,
                do_sample=True,
                max_new_tokens=150,
                num_return_sequences=num_samples,
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

