# Description: Probing uncertainty estimation class.
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity

from src.uncertainty.entailment import get_entailment_score


class ProbingUncertaintyEstimator:
    """ Probing uncertainty estimation class. """
    def __init__(self, model, tokenizer, original_answers):
        self.model = model
        self.tokenizer = tokenizer
        self.original_answers = original_answers

    def compute_uncertainty(self, perturbed_list, similarity_method='cosine') -> float:
        """ Compute uncertainty for a given input and perturbed samples. """
        similarity_score = []
        for sample in perturbed_list[1:]:
            similarity_score.append(self._explanation_similarity(self.original_answers, sample, similarity_method))
        return sum(similarity_score) / len(similarity_score)

    def combine_uncertainties(self, temperature_samples, trigger_samples, rephrase_samples):
        """ Combine uncertainties from different perturbations. """
        return self.compute_uncertainty(temperature_samples) + self.compute_uncertainty(trigger_samples) + self.compute_uncertainty(rephrase_samples)

    def _explanation_similarity(self, text1, text2, method='cosine'):
        """ Passed explanations are split into separate steps and the cross similarity between all steps is computed."""
        # TODO: Possible different ways of computing the similarity between the explanations:
        # 1. Compute the similarity between the entire explanations
        # 2. Compute the similarity between each corresponding step of the explanations (have to add method for when diff lengths
        # 3. Compute the similarity between each step of original explanation and all steps of the second (per step uncertainty)

        text1_steps = text1.split("\n")
        text2_steps = text2.split("\n")

        similarity = 0
        for step1, step2 in zip(text1_steps, text2_steps):
            # Remove the confidence part from each step
            step1_cleaned = self._remove_confidence(step1)
            step2_cleaned = self._remove_confidence(step2)

            # Compute semantic similarity on the cleaned steps
            similarity += self._sematic_similarity(step1_cleaned, step2_cleaned, method)

        # Return the average similarity
        return similarity / len(text1_steps)

    @staticmethod
    def _remove_confidence(step):
        """
        Remove the 'Confidence: [number]' part from the step.
        """
        # Split on "Confidence:" and take the first part
        return step.split("Confidence:")[0].strip()


    def _sematic_similarity(self, text1, text2, method='cosine'):
        # TODO: Change the similarity to be step to step instead of entire explanation (too broad)
        if method == 'cosine':
            return cosine_similarity(text1, text2)
        if method == 'jaccard':
            return jaccard_similarity(text1, text2)
        if method == 'entailment':
            return get_entailment_score(text1, text2)
        else:
            raise NotImplementedError

    def estimate_uncertainty(self, temperature_samples = None, trigger_samples = None, rephrase_samples = None, similarity_method : str = 'cosine'):
        uncertainty = []
        if temperature_samples:
            temperature_uncertainty = self.compute_uncertainty(temperature_samples, similarity_method)
            uncertainty.append(temperature_uncertainty)
        if trigger_samples:
            trigger_uncertainty = self.compute_uncertainty(trigger_samples, similarity_method)
            uncertainty.append(trigger_uncertainty)
        if rephrase_samples:
            rephrase_uncertainty = self.compute_uncertainty(rephrase_samples, similarity_method)
            uncertainty.append(rephrase_uncertainty)
        if uncertainty:
            return sum(uncertainty) / len(uncertainty)
        else:
            raise ValueError("No perturbed samples provided to estimate uncertainty.")

