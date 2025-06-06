import pandas as pd
from src.uncertainty.probing_uncertainty import ProbingUncertaintyEstimator
from src.utils.log_functions import log_message
from src.utils.parsers import parse_arguments_evaluation
import re


def prepare_samples(generated_answers):
    """ Separate samples by perturbation type. """
    temperature_samples = []
    trigger_samples = []
    rephrase_samples = []  # Prep for rephrased samples

    original_answer = generated_answers['original_answer']
    for key, samples in generated_answers.items():
        if key.startswith("temp_"):
            temperature_samples.extend(samples)
        elif key.startswith("trigger_"):
            trigger_samples.extend(samples)
        elif key.startswith("rephrased_"):
            rephrase_samples.extend(samples)

    return temperature_samples, trigger_samples, rephrase_samples, original_answer


def compute_uncertainty_for_row(row) -> dict:
    """Compute and return optimized uncertainty data structure"""
    generated_answers = row["generated_answers"]
    temperature_samples, trigger_samples, rephrase_samples, original_answer = prepare_samples(generated_answers)

    estimator = ProbingUncertaintyEstimator(original_answer)
    uncertainty = estimator.estimate_uncertainty(
        temperature_samples=temperature_samples,
        trigger_samples=trigger_samples,
        rephrase_samples=rephrase_samples
    )

    # Extract confidence from original answer
    original_confidence = extract_confidence(original_answer) if original_answer else 0.5

    # Check correctness (assuming expected_output exists in row)
    original_correct = (extract_final_answer(original_answer) == row["expected_output"]) if original_answer else False

    return {
        "idx": row.name,  # Assuming DataFrame index is question ID
        "input": row["input"],
        "expected_output": row["Answer"],
        "original_answer": original_answer,
        "original_confidence": original_confidence,
        "original_correct": original_correct,
        "uncertainty": uncertainty,
        "perturbed_answers": {
            "temperature": [
                {
                    "answer": extract_final_answer(ans),
                    "confidence": extract_confidence(ans),
                    "correct": extract_final_answer(ans) == row["expected_output"]
                }
                for ans in temperature_samples
            ],
            "trigger": [
                {
                    "answer": extract_final_answer(ans),
                    "confidence": extract_confidence(ans),
                    "correct": extract_final_answer(ans) == row["expected_output"]
                }
                for ans in rephrase_samples
            ],
            "rephrase": [
                {
                    "answer": extract_final_answer(ans),
                    "confidence": extract_confidence(ans),
                    "correct": extract_final_answer(ans) == row["expected_output"]
                }
                for ans in rephrase_samples
            ]
        }
    }

def extract_final_answer(text):
    """Extract the final numerical answer from model output text"""
    # Priority 1: Look for "Final Answer" pattern
    final_answer_match = re.search(r'Final Answer[^:]*:\s*(\d+)', text, re.IGNORECASE)
    if final_answer_match:
        return int(final_answer_match.group(1))

    # Priority 2: Find last equation result
    equations = re.findall(r'(\d+\s*[+\-*/]\s*\d+\s*=\s*\d+)', text)
    if equations:
        last_eq = equations[-1].split('=')[-1].strip()
        return int(last_eq)

    # Priority 3: Find last standalone number (fallback)
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        return int(numbers[-1])

    return None

def extract_confidence(text):
    """Extract confidence percentage from model output text"""
    confidence_match = re.search(r'Overall Confidence(?:\(0-100\))?[^:]*:\s*\d+,\s*(\d+)%', text, re.IGNORECASE)
    return int(confidence_match.group(1)) / 100 if confidence_match else None  # Default to 0.5 if not found


def main(executor: str = "habrok", task: str = "SVAMP", model: str = "gemma9b", index: int =None):

    log_message(f"Starting execution with parameters: executor={executor}, task={task}, model={model}")

    if executor == "habrok":
        result_dir = '/home2/s4637577/thesis-project/results'
    else:
        result_dir = r"C:\Users\DELL\OneDrive\Dokumenty\studia\AI\Year3\ThesisAI\thesis-project\results"

    if index:
        # Load a specific index
        df = pd.read_json(f"{result_dir}/{task}_perturbed_outputs_{model}_{index}_{index + 100}.json")
        df["uncertainty"] = df.apply(compute_uncertainty_for_row, axis=1)
        output_dir = f"{result_dir}/{task}_perturbed_outputs_{model}_{index}_uncertainty.json"
        df.to_json(output_dir, orient="records")
        log_message(f"Finished execution with parameters: index={index}, task={task}, model={model}")
        log_message(f"Results saved to {output_dir}")
    else:
        dataframes = [pd.read_json(f"{result_dir}/{task}_perturbed_outputs_{model}_{i}_{i + 100}.json") for i in
                      range(0, 1000, 100)]
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df["uncertainty"] = combined_df.apply(compute_uncertainty_for_row, axis=1)
        output_dir = f"{result_dir}/{task}_perturbed_outputs_{model}_uncertainty.json"
        combined_df.to_json(output_dir, orient="records")
        log_message(f"Finished execution with parameters: task={task}, model={model}")
        log_message(f"Results saved to {output_dir}")


if __name__ == "__main__":
    args = parse_arguments_evaluation()

    main(executor=args.executor, task=args.task, model=args.model, index=args.index,)
