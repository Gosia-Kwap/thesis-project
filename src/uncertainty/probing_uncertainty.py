from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import os
from dotenv import load_dotenv
from typing import List, Optional
from src.utils.log_functions import log_message
from src.utils.Enums import LEVEL


load_dotenv()


class ProbingUncertaintyEstimator:
    """Enhanced uncertainty estimation using explanation consistency with embedded semantic similarity."""

    def __init__(self, original_answer: str):
        """
        Initialize with original answer and load models.

        Args:
            original_answer: The original model answer to compare against
        """
        self.original_answer = original_answer
        self.token = os.getenv("HUGGING_FACE_TOKEN")

        # Load models
        self._load_models()

        # Pre-process original answer
        self.original_steps = self._preprocess_steps(original_answer)

    def _load_models(self):
        """Load both classification and embedding models."""
        log_message("Loading models for entailment and embeddings...", LEVEL.INFO)
        model_name = "potsawee/deberta-v3-large-mnli"

        # For entailment classification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=self.token)
        self.classification_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, token=self.token
        )
        log_message("Models loaded successfully", LEVEL.INFO)

        # For embeddings (using base DeBERTa)
        self.embedding_model = AutoModel.from_pretrained(model_name, token=self.token)

        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classification_model.to(self.device)
        self.embedding_model.to(self.device)

    def _preprocess_steps(self, explanation: str) -> List[str]:
        """Split explanation into steps and clean them."""
        return [self._remove_confidence(step) for step in explanation.split("\n") if step.strip()]

    @staticmethod
    def _remove_confidence(step: str) -> str:
        """Remove confidence annotations from steps."""
        return step.split("Confidence:")[0].strip()

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from DeBERTa model."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Mean pooling
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

        return embeddings

    def _get_entailment_score(self, text1: str, text2: str, return_probs: bool = False):
        """
        Compute entailment between two texts.

        Args:
            return_probs: If True, returns raw probabilities instead of discrete score
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.classification_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        if return_probs:
            return {
                "contradiction": probs[0].item(),
                "neutral": probs[1].item(),
                "entailment": probs[2].item()
            }
        else:
            return torch.argmax(logits, dim=1).item() - 1  # Scale to [-1, 0, 1]

    def _semantic_similarity(self, text1: str, text2: str, method: str = 'cosine') -> float:
        """
        Compute semantic similarity between two texts using specified method.

        Available methods:
        - 'cosine': Cosine similarity between embeddings
        - 'jaccard': Jaccard similarity on binarized embeddings
        - 'entailment': Entailment score (-1, 0, 1)
        - 'entailment_prob': Entailment probability (0-1)
        """
        if method.startswith('entailment'):
            if method == 'entailment_prob':
                probs = self._get_entailment_score(text1, text2, return_probs=True)
                return probs['entailment'] - probs['contradiction']  # Scale to [-1, 1]
            return self._get_entailment_score(text1, text2)

        # Get embeddings for both texts
        emb1, emb2 = self._get_embeddings([text1, text2])

        if method == 'cosine':
            return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        elif method == 'jaccard':
            # Convert embeddings to binary vectors for Jaccard
            binary1 = (emb1 > np.mean(emb1)).astype(int)
            binary2 = (emb2 > np.mean(emb2)).astype(int)
            return jaccard_score(binary1.flatten(), binary2.flatten())
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def _step_similarity(self, steps1: List[str], steps2: List[str], method: str = 'cosine') -> float:
        """
        Compute average similarity between aligned steps.
        Uses optimal alignment when step counts differ.
        """
        if not steps1 or not steps2:
            return 0.0

        # Get all pairwise similarities
        similarities = []
        for step1 in steps1:
            for step2 in steps2:
                similarities.append(self._semantic_similarity(step1, step2, method))

        # Find best matches when counts differ
        if len(steps1) != len(steps2):
            # Create similarity matrix
            sim_matrix = np.array(similarities).reshape(len(steps1), len(steps2))
            # Use maximum similarities
            return max(sim_matrix.max(axis=0).mean(), sim_matrix.max(axis=1).mean())

        return np.mean(similarities[:min(len(steps1), len(steps2))])

    def compute_uncertainty(self, perturbed_samples: List[str], method: str = 'cosine') -> float:
        """
        Compute uncertainty by comparing original answer to perturbed samples.

        Args:
            perturbed_samples: List of perturbed answers
            method: Similarity method to use
        """
        if not perturbed_samples:
            return 0.0

        similarities = []
        original_steps = self.original_steps

        for sample in perturbed_samples:
            sample_steps = self._preprocess_steps(sample)
            similarity = self._step_similarity(original_steps, sample_steps, method)
            similarities.append(similarity)

        # Uncertainty is 1 minus average similarity
        return 1 - (sum(similarities) / len(similarities))

    def estimate_uncertainty(
            self,
            temperature_samples: Optional[List[str]] = None,
            trigger_samples: Optional[List[str]] = None,
            rephrase_samples: Optional[List[str]] = None,
            method: str = 'cosine',
            weights: Optional[List[float]] = None
    ) -> float:
        """
        Combined uncertainty estimate from different perturbation types.

        Args:
            weights: Optional weights for [temperature, trigger, rephrase] uncertainties
        """
        uncertainties = []

        if temperature_samples:
            uncertainties.append(self.compute_uncertainty(temperature_samples, method))
        if trigger_samples:
            uncertainties.append(self.compute_uncertainty(trigger_samples, method))
        if rephrase_samples:
            uncertainties.append(self.compute_uncertainty(rephrase_samples, method))

        if not uncertainties:
            raise ValueError("No perturbed samples provided")

        if weights and len(weights) == len(uncertainties):
            return sum(w * u for w, u in zip(weights, uncertainties)) / sum(weights)
        return sum(uncertainties) / len(uncertainties)