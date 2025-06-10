import pandas as pd
from src.uncertainty.probing_uncertainty import ProbingUncertaintyEstimator
from src.utils.log_functions import log_message
from src.utils.parsers import parse_arguments_evaluation
import re

task = 'SVAMP'
method = 'cosine'
model = 'gemma9b'

result_dir = r"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/uncertainty"
for i in range(200, 300, 100):
    dataframes = pd.read_json(f"{result_dir}/{task}_perturbed_outputs_{model}_{i}_uncertainty_{method}.json")
    results = dataframes['uncertainty']
    output_dir = f"{result_dir}/{task}_perturbed_outputs_{model}_{i}_uncertainty_{method}.json"
    results.to_json(output_dir, orient="records")
