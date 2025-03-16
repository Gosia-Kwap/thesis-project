# Description: Probing uncertainty estimation class.
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity

from src.uncertainty.entailment import get_entailment_score


class ProbingUncertaintyEstimator:
    """ Probing uncertainty estimation class. """
    def __init__(self, model, tokenizer, temperature_range, trigger_phrases, rephrase_strategies, original_answers):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature_range = temperature_range
        self.trigger_phrases = trigger_phrases
        self.rephrase_strategies = rephrase_strategies
        self.original_answers = original_answers

    def temperature_perturbation(self, input_text, num_samples):
        """ Generate perturbed samples using temperature scaling. """
        perturbed_samples = []
        for temperature in self.temperature_range:
            perturbed_samples.extend(self._temperature_perturbation(input_text, temperature, num_samples))
        return perturbed_samples

    def trigger_phrase_perturbation(self, input_text, num_samples):
        """ Generate perturbed samples using trigger phrases. """
        perturbed_samples = []
        for trigger_phrase in self.trigger_phrases:
            perturbed_samples.extend(self._trigger_phrase_perturbation(input_text, trigger_phrase, num_samples))
        return perturbed_samples

    def rephrase_perturbation(self, input_text, num_samples):
        perturbed_samples = []
        for rephrase_strategy in self.rephrase_strategies:
            perturbed_samples.extend(self._rephrase_perturbation(input_text, rephrase_strategy, num_samples))
        return perturbed_samples

    def compute_uncertainty(self, perturbed_list, similarity_method='cosine') -> float:
        """ Compute uncertainty for a given input. """
        similarity_score = []
        for sample in perturbed_list[1:]:
            similarity_score.append(self.sematic_similarity(perturbed_list[0], sample, similarity_method))
        return sum(similarity_score) / len(similarity_score)



    def combine_uncertainties(self, temperature_samples, trigger_samples, rephrase_samples):
        return self.compute_uncertainty(temperature_samples) + self.compute_uncertainty(trigger_samples) + self.compute_uncertainty(rephrase_samples)

    def sematic_similarity(self, text1, text2, method='cosine'):
        if method == 'cosine':
            return cosine_similarity(text1, text2)
        if method == 'jaccard':
            return jaccard_similarity(text1, text2)
        if method == 'entailment':
            return get_entailment_score(text1, text2)
        else:
            raise NotImplementedError

    def estimate_uncertainty(self, input_text, num_samples, similarity_method : str = 'cosine'):
        temperature_samples = self.temperature_perturbation(input_text, num_samples)
        trigger_samples = self.trigger_phrase_perturbation(input_text, num_samples)
        rephrase_samples = self.rephrase_perturbation(input_text, num_samples)
        return self.combine_uncertainties(temperature_samples, trigger_samples, rephrase_samples)

    def _temperature_perturbation(self, input_text, temperature, num_samples):
        perturbed_samples = []
        for _ in range(num_samples):
            perturbed_samples.append(self._generate_perturbed_sample(input_text, temperature))
        return perturbed_samples

    def _trigger_phrase_perturbation(self, input_text, trigger_phrase, num_samples):
        perturbed_samples = []
        for _ in range(num_samples):
            perturbed_samples.append(self._generate_perturbed_sample(input_text, trigger_phrase))
        return perturbed_samples

    def _rephrase_perturbation(self, input_text, rephrase_strategy, num_samples):
        perturbed_samples = []
        for _ in range(num_samples):
            perturbed_samples.append(self._generate_perturbed_sample(input_text, rephrase_strategy))
        return perturbed_samples
