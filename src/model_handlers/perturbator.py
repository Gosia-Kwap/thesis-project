from prompts.CoT import generic_prompt, trigger_phrases

class Perturbator:
    def __init__(self, model):
        self.model = model
        self.default_temp = {'gpt': 0.7, 'gpt2': 0.7, 'gpt3': 0.7}
        self.prompt = generic_prompt
        self.trigger_phrases = trigger_phrases

    def temperature_perturbation(self, task, input_text,  num_samples):
        """ Generate perturbed samples using temperature scaling. """
        default_temp = self.default_temp[self.model]
        temperatures = [default_temp / 2,  default_temp, default_temp + (1 - default_temp) / 2, 1]
        perturbed_samples = []
        for temperature in temperatures:
            perturbed_samples.extend(self._generate_samples(task, input_text, num_samples, temperature=temperature))
        return perturbed_samples

    def trigger_phrase_perturbation(self, task, input_text, num_samples):
        """ Generate perturbed samples using trigger phrases. """
        perturbed_samples = []
        for trigger_phrase in trigger_phrases:
            perturbed_samples.extend(self._generate_samples(task, input_text, num_samples, trigger_phrase=trigger_phrase))
        return perturbed_samples

    def _generate_samples(self, task, input_text, num_samples, temperature=None, trigger_phrase=None):
        if trigger_phrase:
            prompt = trigger_phrase + self.prompt
        else:
            prompt = self.trigger_phrases[0] + self.prompt

        if temperature is None:
            temperature = self.default_temp.get(self.model.config.model_type, 0.7)

        # Generate multiple sequences in one call
        generated_texts = self.model.generate(
            input_ids=self.model.tokenizer.encode(prompt + input_text, return_tensors='pt'),
            temperature=temperature,
            num_return_sequences=num_samples
        )

        # Decode and return the generated texts
        return [self.model.tokenizer.decode(text, skip_special_tokens=True) for text in generated_texts]