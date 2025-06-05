import pandas as pd
from src.uncertainty.probing_uncertainty import ProbingUncertaintyEstimator
from src.utils.log_functions import log_message
from src.utils.parsers import parse_arguments_evaluation


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


def compute_uncertainty_for_row(row):
    """ Compute uncertainty for a single DataFrame row. """
    generated_answers = row["generated_answers"]

    temperature_samples, trigger_samples, rephrase_samples, original_answer = prepare_samples(generated_answers)

    # Instantiate the estimator
    estimator = ProbingUncertaintyEstimator(original_answer)

    # Estimate uncertainty
    uncertainty = estimator.estimate_uncertainty(
        temperature_samples=temperature_samples,
        trigger_samples=trigger_samples,
        rephrase_samples=rephrase_samples
    )

    return uncertainty


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
