import pandas as pd
from src.uncertainty.probing_uncertainty import ProbingUncertaintyEstimator
from src.utils.log_functions import log_message
from src.utils.parsers import parse_arguments_evaluation
import re

import re


def extract_final_answer(text: str, answer_type: str = 'integer'):
    """
    Extract the final answer from model output text with configurable return type.

    Args:
        text: The model output text to parse
        answer_type: The type of answer to extract ('integer', 'float', or 'label')

    Returns:
        The extracted answer in the requested format, or None if no match found
    """
    # Pre-process text to remove common noise
    clean_text = re.sub(r'\s+', ' ', text.strip())

    # Try explicit answer patterns first (highest priority)
    patterns = [
        r'Final Answer[^:]*:\s*[\$€£]?\s*([^\n.]+)',  # $123 or €45 etc.
        r'final answer is[:\s]*([^\n.]+)',
        r'answer is[:\s]*([^\n.]+)',
        r'[Tt]he answer is\s*([^\n.]+)',
        r'[Tt]herefore\s*(?:the|our)\s*answer is\s*([^\n.]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            return _convert_to_type(extracted, answer_type)

    # Special handling for multiple choice labels (A/B/C/D)
    if answer_type == 'label':
        label_match = re.search(r'\b([ABCD])\b(?!\.\d)', clean_text, re.IGNORECASE)
        if label_match:
            return label_match.group(1).upper()

    # Find equations (medium priority)
    equation = re.search(r'\bx\s*=\s*([^\n\r]+)', clean_text, re.IGNORECASE)
    if equation:
        return _convert_to_type(equation.group(1).strip(), answer_type)

    # Find last number that is NOT a confidence percentage (low priority)
    if answer_type in ['integer', 'float']:
        numbers = re.findall(r'(\d+(?:\.\d+)?)(?!\s*%)', clean_text)
        if numbers:
            return _convert_to_type(numbers[-1].strip(), answer_type)

    # Final fallback - find any number
    if answer_type in ['integer', 'float']:
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', clean_text)
        if numbers:
            return _convert_to_type(numbers[-1].strip(), answer_type)

    return None


def _convert_to_type(value: str, target_type: str):
    """Convert extracted string to the desired type"""
    try:
        if target_type == 'integer':
            # Remove any non-numeric characters before conversion
            clean_value = re.sub(r'[^\d-]', '', value)
            return int(float(clean_value))  # Handle cases like "3.0"
        elif target_type == 'float':
            # Handle currency symbols, commas, etc.
            clean_value = re.sub(r'[^\d.-]', '', value)
            return float(clean_value)
        elif target_type == 'label':
            # Return uppercase version if it's A/B/C/D
            if value.upper() in ['A', 'B', 'C', 'D']:
                return value.upper()
            return value
        return value
    except (ValueError, TypeError):
        return value  # Return as-is if conversion fails


def extract_confidence(text):
    """Extract confidence percentage from model output text."""
    match = re.search(r'Overall Confidence(?:\(0-100\))?[^:]*:\s*\$?\d+,\s*(\d+)%', text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100

    # If no match, find last time there was a confidence
    pattern = r'confidence\s*(?:is)?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%'
    fallback_matches = re.findall(pattern, text, re.IGNORECASE)
    if fallback_matches:
        return float(fallback_matches[-1])

    # If no match, find last number with percentage
    percent_numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    if percent_numbers:
        return float(percent_numbers[-1])

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
            "answer": (ans_val := extract_final_answer(ans)),
            "confidence": extract_confidence(ans),
            "correct": ans_val == expected_output
        }
        for ans in samples
    ]


def compute_uncertainty_for_row(row, method = 'cosine') -> dict:
    """Compute structured uncertainty metrics for a single row."""
    temp, trigger, rephrase, original = prepare_samples(row["generated_answers"])

    estimator = ProbingUncertaintyEstimator(original)
    uncertainty = estimator.estimate_uncertainty(temp, trigger, rephrase, method=method)

    original_val = extract_final_answer(original)
    original_conf = extract_confidence(original)

    return {
        "idx": row.name,
        "expected_output": row["expected_output"],
        "full_original_answer": original,
        "original_answer": original_val,
        "original_confidence": original_conf,
        "original_correct": original_val == row["expected_output"],
        "uncertainty": uncertainty,
        "perturbed_answers": {
            "temperature": build_perturbed_set(temp, row["expected_output"]),
            "trigger": build_perturbed_set(trigger, row["expected_output"]),
            "rephrase": build_perturbed_set(rephrase, row["expected_output"])
        }
    }


def main(executor: str = "habrok", task: str = "SVAMP", model: str = "gemma9b", index: int =None, method: str = "cosine"):

    log_message(f"Starting execution with parameters: executor={executor}, task={task}, model={model}")

    if executor == "habrok":
        result_dir = '/home2/s4637577/thesis-project/results'
    else:
        result_dir = r"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results"

    if index:
        # Load a specific index
        df = pd.read_json(f"{result_dir}/{task}_perturbed_outputs_{model}_{index}_{index + 100}.json")
        results = df.apply(lambda row: compute_uncertainty_for_row(row, method), axis=1)
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
        f"  Range: {args.index}-{args.index + 100}\n"
    )

    main(executor=args.executor, task=args.task, model=args.model, index=args.index, method=args.method)
