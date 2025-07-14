import sys
import time
print("[DEBUG] Python entrypoint reached", file=sys.stderr, flush=True)
time.sleep(0.5)

import pandas as pd
from src.uncertainty.probing_uncertainty import ProbingUncertaintyEstimator
from src.utils.log_functions import log_message
from src.utils.parsers import parse_arguments_evaluation
from src.utils.Enums import format_dict

import re


def extract_final_answer(text, expected_type="int"):
    """
    Extract final answer from text and convert to expected type.

    expected_type: "float" | "int" | "label"
    """
    patterns = [
        r'Final Answer and Overall Confidence\(0-100\):\s*([A-Ea-e]|\S*\d+(?:\.\d+)?\S*)\s*,\s*\d+%',
        r'final answer[^:]*:\s*([A-Ea-e]|\S*\d+(?:\.\d+)?\S*)',
        r'answer is[:\s]*([A-Ea-e]|\S*\d+(?:\.\d+)?\S*)',
        r'\bx\s*=\s*([A-Ea-e]|\S*\d+(?:\.\d+)?\S*)'
    ]

    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return convert_answer(match.group(1), expected_type)

    # fallback: last number with optional surrounding units, not a %
    numbers = re.findall(r'(\S*\d+(?:\.\d+)?\S*)(?!\s*%)', text)
    if numbers:
        return convert_answer(numbers[-1], expected_type)

    return None


def convert_answer(ans_str, expected_type):
    """Convert ans_str to desired type: float, int, or label."""
    ans_str = ans_str.strip()

    if expected_type == "label":
        # Expect A-E letter
        if re.fullmatch(r'[A-Ea-e]', ans_str):
            return ans_str.upper()
        else:
            return ans_str  # return as-is

    # Remove non-digit, non-dot, non-minus at edges (strip units)
    numeric_match = re.search(r'-?\d+(?:\.\d+)?', ans_str)
    if not numeric_match:
        return ans_str  # fallback if no number found

    num_str = numeric_match.group(0)

    try:
        if expected_type == "int":
            return int(float(num_str))
        elif expected_type == "float":
            return float(num_str)
    except ValueError:
        return ans_str  # fallback: couldn't convert

    return ans_str



def extract_confidence(text):
    """Extract confidence percentage from model output text."""
    match = re.search(r'Overall Confidence(?:\(0-100\))?[^:]*:\s*\$?\d+,\s*(\d+)%', text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100

    # If no match, find last time there was a confidence
    pattern = r'confidence\s*(?:is)?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%'
    fallback_matches = re.findall(pattern, text, re.IGNORECASE)
    if fallback_matches:
        return float(fallback_matches[-1]) / 100

    # If no match, find last number with percentage
    percent_numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    if percent_numbers:
        return float(percent_numbers[-1]) / 100

    return None


def prepare_samples(generated):
    """Separate samples by perturbation type."""
    temp = [s for k, v in generated.items() if k.startswith("temp_") for s in v]
    trigger = [s for k, v in generated.items() if k.startswith("trigger_") for s in v]
    rephrase = [s for k, v in generated.items() if k.startswith("rephrased_") for s in v]
    return temp, trigger, rephrase, generated.get("original_answer", [None])[0]


def build_perturbed_set(samples: list[str], expected_output, answer_format):
    """Create metadata for a list of perturbed samples."""
    return [
        {
            "answer": (ans_val := extract_final_answer(ans, answer_format)),
            "confidence": extract_confidence(ans),
            "correct": ans_val == expected_output
        }
        for ans in samples
    ]


def compute_uncertainty_for_row(row, method = 'cosine', answer_format:str = 'int') -> dict:
    """Compute structured uncertainty metrics for a single row."""
    temp, trigger, rephrase, original = prepare_samples(row["generated_answers"])

    original_val = extract_final_answer(original, answer_format)
    original_conf = extract_confidence(original)

    estimator = ProbingUncertaintyEstimator(original)
    uncertainty = estimator.estimate_uncertainty(temp, trigger, rephrase, method=method)

    return {
        "idx": row.name,
        "expected_output": row["expected_output"],
        "full_original_answer": original,
        "original_answer": original_val,
        "original_confidence": original_conf,
        "original_correct": original_val == row["expected_output"],
        "uncertainty": uncertainty,
        "perturbed_answers": {
            "temperature": build_perturbed_set(temp, row["expected_output"], answer_format),
            "trigger": build_perturbed_set(trigger, row["expected_output"], answer_format),
            "rephrase": build_perturbed_set(rephrase, row["expected_output"], answer_format)
        }
    }


def main(executor: str = "habrok", task: str = "SVAMP", model: str = "gemma9b", index: int =None, method: str = "cosine", quantisation: str = None):

    log_message(f"Starting execution with parameters: executor={executor}, task={task}, model={model}, quantisation={quantisation}")

    if executor == "habrok":
        result_dir = '/home2/s4637577/thesis-project/results'
    else:
        result_dir = r"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/results-newest"

    suffix = f"_{quantisation}" if quantisation else ""

    if index is not None:
        # Load a specific index
        df = pd.read_json(f"{result_dir}/{task}_perturbed_outputs_{model}_{index}_{index + 100}{suffix}.json")
        results = df.apply(lambda row: compute_uncertainty_for_row(row, method=method, answer_format=format_dict[task]), axis=1)
        output_dir = f"{result_dir}/uncertainty/{task}_perturbed_outputs_{model}_{index}_uncertainty_{method}.json"
        results.to_json(output_dir, orient="records")
        log_message(f"Finished execution with parameters: index={index}, task={task}, model={model}")
        log_message(f"Results saved to {output_dir}")
    else:
        dataframes = [pd.read_json(f"{result_dir}/{task}_perturbed_outputs_{model}_{i}_{i + 100}.json") for i in
                      range(0, 1000, 100)]
        combined_df = pd.concat(dataframes, ignore_index=True)
        results = combined_df.apply(lambda row: compute_uncertainty_for_row(row, method), axis=1)
        output_dir = f"{result_dir}/{task}_perturbed_outputs_{model}_uncertainty_{method}.json"
        results.to_json(output_dir, orient="records")
        log_message(f"Finished execution with parameters: task={task}, model={model}")
        log_message(f"Results saved to {output_dir}")


if __name__ == "__main__":
    args = parse_arguments_evaluation()

    log_message(
        f"Starting uncertainty evaluation with parameters:\n"
        f"  Model: {args.model}\n"
        f"  Method: {args.method}\n"
        f"  Task: {args.task}\n"
        f"  Range: {args.index}-{args.index +100}\n" if args.index else None,
        f"  Quantisation: {args.quantisation}\n"
    )

    main(executor=args.executor, task=args.task, model=args.model, index=args.index, method=args.method, quantisation=args.quantisation)
