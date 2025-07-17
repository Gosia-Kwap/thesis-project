from src.uncertainty.UncertaintyCalibrationAnalyser import UncertaintyCalibrationAnalyser
from src.utils.log_functions import log_message

tasks = ['SVAMP', 'GSM8K', 'ASDiv', 'ai2_arc', 'logiqa', 'CommonsenseQA']
quantisations = ['', '_4', '_6']
methods = ['cosine']
models = ['gemma9b', 'llama3', 'deepseek']
method = methods[0]

def analyse_results(task, model, method, quantisation):
    result_dir = f"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/uncertainty/"
    plot_dir = f"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/plots/{task}/{task}_{method}_{model}{quantisation}"
    stats_dir = f"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/stats/{task}/{model}_{method}{quantisation}"



    # Initialize the analyser with the directory containing results
    analyser = UncertaintyCalibrationAnalyser(
        result_dir=result_dir,
        file_pattern=f"{task}_perturbed_outputs_{model}_*_uncertainty_{method}{quantisation}.json",
        num_bins=10,
        mode="original"  # Use "original" to only analyze the original answers
    )

    # Analyze calibration (confidence and overall uncertainty)
    analysis_results = analyser.analyze_calibration_conf_unc(save_dir=stats_dir)
    #
    # # Plot calibration comparison
    analyser.plot_calibration_unc_conf(analysis_results, plot_dir, mode='uncertainty')

    # Compare uncertainty distributions across types and perturbations
    uncertainty_distributions = analyser.generate_distribution_comparison_per_uncertainty(plot_dir=plot_dir, stats_dir=stats_dir)

    # Perform separate calibration analysis per uncertainty type (overall, temp, trigger, rephrase)
    all_uncertainty_results = analyser.analyze_all_uncertainties(save_path=stats_dir)

    # Plot calibration curves for each uncertainty type
    analyser.plot_calibration_all_unc(all_uncertainty_results, save_dir=plot_dir)
    # Compare results
    print("Confidence ECE:", analysis_results['confidence']['ece'])
    print("Uncertainty ECE:", analysis_results['uncertainty']['ece'])
    print("Accuracy on the dataset:", analysis_results['accuracy'])

    results = {
        "calibration": analysis_results,
        "distribution": uncertainty_distributions,
        "all_uncertainties_calibration": all_uncertainty_results,
    }

    # results_df = pd.DataFrame(results)
    analyser.save_calibration_conf_unc(results, output_path=stats_dir+"_overall_stats.json")

for tasks in tasks:
    for model in models:
        for quantisation in quantisations:
            try:
                log_message(f"Analysis for: {tasks}/{model}{quantisation}/{method}")
                analyse_results(tasks, model, method, quantisation)
            except:
                print(f"Error in analysing {tasks}/{model}{quantisation}/{method}")
                pass


