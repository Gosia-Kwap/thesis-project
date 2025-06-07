import pandas as pd
from src.uncertainty.probing_uncertainty import ProbingUncertaintyEstimator
from src.utils.log_functions import log_message
from src.utils.parsers import parse_arguments_evaluation
import re
import os



def extract_final_answer(text):
    """Extract the final numerical answer from model output text."""
    match = re.search(r'Final Answer[^:]*:\s*\$?(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    equations = re.findall(r'(\d+\s*[+\-*/]\s*\d+\s*=\s*\d+)', text)
    if equations:
        return int(equations[-1].split('=')[-1].strip())

    numbers = re.findall(r'\b\d+\b', text)
    return int(numbers[-1]) if numbers else None


def extract_confidence(text):
    """Extract confidence percentage from model output text."""
    match = re.search(r'Overall Confidence(?:\(0-100\))?[^:]*:\s*\$?\d+,\s*(\d+)%', text, re.IGNORECASE)
    return int(match.group(1)) / 100 if match else None


def prepare_samples(generated):
    """Separate samples by perturbation type."""
    temp = [s for k, v in generated.items() if k.startswith("temp_") for s in v]
    trigger = [s for k, v in generated.items() if k.startswith("trigger_") for s in v]
    rephrase = [s for k, v in generated.items() if k.startswith("rephrased_") for s in v]
    return temp, trigger, rephrase, generated.get("original_answer", [None])[0]


def build_perturbed_set(samples: list[str], expected_output):
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
        result_dir = r"/Users/m.kwapniewska/OneDrive/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results"

    if index:
        # Load a specific index
        df = pd.read_json(f"{result_dir}/{task}_perturbed_outputs_{model}_{index}_{index + 100}.json")
        df["uncertainty"] = df.apply(lambda row: compute_uncertainty_for_row(row, method), axis=1)
        output_dir = f"{result_dir}/uncertainty/{task}_perturbed_outputs_{model}_{index}_uncertainty.json"
        df.to_json(output_dir, orient="records")
        log_message(f"Finished execution with parameters: index={index}, task={task}, model={model}")
        log_message(f"Results saved to {output_dir}")
    else:
        dataframes = [pd.read_json(f"{result_dir}/{task}_perturbed_outputs_{model}_{i}_{i + 100}.json") for i in
                      range(0, 1000, 100)]
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df["uncertainty"] = df.apply(lambda row: compute_uncertainty_for_row(row, method), axis=1)
        output_dir = f"{result_dir}/{task}_perturbed_outputs_{model}_uncertainty.json"
        combined_df.to_json(output_dir, orient="records")
        log_message(f"Finished execution with parameters: task={task}, model={model}")
        log_message(f"Results saved to {output_dir}")


if __name__ == "__main__":
    args = parse_arguments_evaluation()

    main(executor=args.executor, task=args.task, model=args.model, index=args.index, method=args.method)
