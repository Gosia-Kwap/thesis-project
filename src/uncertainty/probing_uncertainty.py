from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Tuple
from src.utils.log_functions import log_message
from src.utils.Enums import LEVEL
import time

load_dotenv()


class ProbingUncertaintyEstimator:
    """Enhanced uncertainty estimation using explanation consistency with embedded semantic similarity."""

    def __init__(self, original_answer: str, log_level: LEVEL = LEVEL.INFO):
        """
        Initialize with original answer and load models.

        Args:
            original_answer: The original model answer to compare against
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.original_answer = original_answer
        self.token = os.getenv("HUGGING_FACE_TOKEN")
        self.log_level = log_level
        self._init_time = time.time()

        # Load models
        self._load_models()

        # Pre-process original answer
        self.original_steps = self._preprocess_steps(original_answer)

    def _load_models(self):
        """Load both classification and embedding models."""
        model_name = "potsawee/deberta-v3-large-mnli"
        start_time = time.time()

        try:
            # For entailment classification
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                token=self.token
            )

            self.classification_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                token=self.token
            )

            self.embedding_model = AutoModel.from_pretrained(
                model_name,
                token=self.token
            )

            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.classification_model.to(self.device)
            self.embedding_model.to(self.device)

        except Exception as e:
            log_message(f"Failed to load models: {str(e)}", LEVEL.ERROR)
            raise

    def _preprocess_steps(self, explanation: str) -> List[str]:
        """Split explanation into steps and clean them."""
        steps = [step.strip() for step in explanation.split("\n") if step.strip()]
        cleaned_steps = [self._remove_confidence(step) for step in steps]

        if self.log_level == LEVEL.DEBUG:
            log_message(f"Preprocessed steps: {cleaned_steps}", self.log_level)

        return cleaned_steps

    @staticmethod
    def _remove_confidence(step: str) -> str:
        """Remove confidence annotations from steps."""
        return step.split("Confidence:")[0].strip()

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from DeBERTa model."""
        if self.log_level==LEVEL.DEBUG:
            log_message(f"Generating embeddings for {len(texts)} texts...", self.log_level)
        start_time = time.time()

        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

            elapsed = time.time() - start_time
            if self.log_level == LEVEL.DEBUG:
                log_message(f"Generated embeddings in {elapsed:.2f}s (shape: {embeddings.shape})",
                        self.log_level)

            return embeddings

        except Exception as e:
            log_message(f"Embedding generation failed: {str(e)}", LEVEL.ERROR)
            raise

    def _get_entailment_score(self, text1: str, text2: str, return_probs: bool = False):
        """
        Compute entailment between two texts.

        Args:
            return_probs: If True, returns raw probabilities instead of discrete score
        """
        if self.log_level == LEVEL.DEBUG:
            log_message(f"Computing entailment between:\n  Text1: {text1[:50]}...\n  Text2: {text2[:50]}...",
                        self.log_level)


        try:
            inputs = self.tokenizer(
                text1,
                text2,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.classification_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]

            result = {
                "contradiction": probs[0].item(),
                "neutral": probs[1].item(),
                "entailment": probs[2].item()
            } if return_probs else torch.argmax(logits, dim=1).item() / 2

            return result

        except Exception as e:
            log_message(f"Entailment computation failed: {str(e)}", LEVEL.ERROR)
            raise

    def _semantic_similarity(self, text1: str, text2: str, method: str = 'cosine') -> float:
        """
        Compute semantic similarity between two texts using specified method.
        """
        if self.log_level == LEVEL.DEBUG:
            log_message(f"Computing {method} similarity between texts...", self.log_level)

        try:
            if method.startswith('entailment'):
                if method == 'entailment_prob':
                    probs = self._get_entailment_score(text1, text2, return_probs=True)
                    result = (probs['entailment'] - probs['contradiction'] + 1) / 2
                else:
                    result = self._get_entailment_score(text1, text2)
            else:
                # Get embeddings for both texts
                emb1, emb2 = self._get_embeddings([text1, text2])

                if method == 'cosine':
                    result = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
                else:
                    raise ValueError(f"Unknown similarity method: {method}")
            if self.log_level == LEVEL.DEBUG:
                log_message(f"{method} similarity result: {result:.3f}", self.log_level)
            return result

        except Exception as e:
            log_message(f"Similarity computation failed: {str(e)}", LEVEL.ERROR)
            raise

    def _step_similarity(self, steps1: List[str], steps2: List[str], method: str = 'cosine') -> float:
        """
        Compute average similarity between aligned steps.
        - If the step lists have equal length: compare step[i] to step[i].
        - If unequal: compare each step in the shorter list to the best-matching step in the longer one.
        """
        if not steps1 or not steps2:
            log_message("Empty steps provided, returning 0 similarity", self.log_level)
            return 0.0

        if self.log_level == LEVEL.DEBUG:
            log_message(f"Computing step similarity ({len(steps1)} vs {len(steps2)} steps)", self.log_level)

        start_time = time.time()

        try:
            if len(steps1) == len(steps2):
                # Aligned step-to-step comparison
                similarities = [
                    self._semantic_similarity(s1, s2, method)
                    for s1, s2 in zip(steps1, steps2)
                ]
                result = np.mean(similarities)
            else:
                # Greedy matching shorter list aligned to best matches in longer list
                short, long = (steps1, steps2) if len(steps1) < len(steps2) else (steps2, steps1)
                similarities = []
                for s1 in short:
                    best_sim = max(
                        self._semantic_similarity(s1, s2, method)
                        for s2 in long
                    )
                    similarities.append(best_sim)
                result = np.mean(similarities)

            elapsed = time.time() - start_time
            if self.log_level == LEVEL.DEBUG:
                log_message(f"Step similarity computed in {elapsed:.2f}s: {result:.3f}", self.log_level)

            return result

        except Exception as e:
            log_message(f"Step similarity computation failed: {e}", self.log_level)
            return 0.0

    def compute_uncertainty(self, perturbed_samples: List[str], method: str = 'cosine') -> float:
        """
        Compute uncertainty by comparing original answer to perturbed samples.
        """
        if not perturbed_samples:
            log_message("No perturbed samples provided, returning 0 uncertainty", self.log_level)
            return 0.0

        start_time = time.time()
        similarities = []

        for i, sample in enumerate(perturbed_samples):
            sample_steps = self._preprocess_steps(sample)
            similarity = self._step_similarity(self.original_steps, sample_steps, method)
            similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities)
        uncertainty = 1 - avg_similarity


        return uncertainty

    def estimate_uncertainty(
            self,
            temperature_samples: Optional[List[str]] = None,
            trigger_samples: Optional[List[str]] = None,
            rephrase_samples: Optional[List[str]] = None,
            method: str = 'cosine',
    ) -> Dict[str, float]:
        """
        Compute average uncertainty per perturbation method and overall average.
        Returns a dictionary with keys: 'temp', 'trigger', 'rephrase', 'overall'.
        """
        uncertainties = {}

        if temperature_samples:
            temp_uncertainty = self.compute_uncertainty(temperature_samples, method)
            uncertainties['temp'] = np.mean(temp_uncertainty)

        if trigger_samples:
            trigger_uncertainty = self.compute_uncertainty(trigger_samples, method)
            uncertainties['trigger'] = np.mean(trigger_uncertainty)

        if rephrase_samples:
            rephrase_uncertainty = self.compute_uncertainty(rephrase_samples, method)
            uncertainties['rephrase'] = np.mean(rephrase_uncertainty)

        if not uncertainties:
            raise ValueError("No perturbed samples provided")

        # Compute overall average across available methods
        uncertainties['overall'] = np.mean(list(uncertainties.values()))

        return uncertainties