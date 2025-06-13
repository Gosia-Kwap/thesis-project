import pandas as pd
from src.uncertainty.probing_uncertainty import ProbingUncertaintyEstimator
from src.utils.log_functions import log_message
from src.utils.parsers import parse_arguments_evaluation
import re

task = 'SVAMP'
method = 'cosine'
model = 'gemma9b'

result_dir = r"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/uncertainty"
for i in range(100, 300, 100):
    dataframes = pd.read_json(f"{result_dir}/{task}_perturbed_outputs_{model}_{i}_uncertainty_{method}.json")
    results = dataframes['uncertainty']
    original_answers = dataframes['generated_answers'].apply(lambda d: d.get('original_answer', ''))
    results_with_original = results.copy()
    results_with_original[:] = [
        {**d, 'original_answer_text': oa} for d, oa in zip(results, original_answers)
    ]

    output_dir = f"{result_dir}/{task}_perturbed_outputs_{model}_{i}_uncertainty_{method}.json"
    results_with_original.to_json(output_dir, orient="records")
