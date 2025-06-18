from src.uncertainty.UncertaintyCalibrationAnalyser import UncertaintyCalibrationAnalyser

task = "SVAMP"
model = 'llama3'
method = 'cosine'
result_dir = "/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/uncertainty"
plot_dir = f"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/plots/{task}/"


# Initialize the analyser with the directory containing results
analyser = UncertaintyCalibrationAnalyser(
    result_dir=result_dir,
    file_pattern=f"{task}_perturbed_outputs_{model}_*_uncertainty_{method}.json",
    num_bins=15,
    mode="original"  # Use "original" to only analyze the original answers
)

# Analyze calibration (confidence and overall uncertainty)
analysis_results = analyser.analyze_calibration_conf_unc()

# Plot calibration comparison
analyser.plot_calibration_unc_conf(analysis_results, f"{plot_dir}_{model}_calibration.png")

# Optional: Save results to JSON
# analyser.save_results(analysis_results['confidence'], 'confidence_results.json')
# analyser.save_results(analysis_results['uncertainty'], 'uncertainty_results.json')

# Compare uncertainty distributions across types and perturbations
uncertainty_distributions = analyser.generate_distribution_comparison_per_uncertainty(save_dir=plot_dir, model=model)

# Perform separate calibration analysis per uncertainty type (overall, temp, trigger, rephrase)
all_uncertainty_results = analyser.analyze_all_uncertainties()

# Plot calibration curves for each uncertainty type
analyser.plot_calibration_all_unc(all_uncertainty_results, save_dir=plot_dir)

# Compare results
print("Confidence ECE:", analysis_results['confidence']['ece'])
print("Uncertainty ECE:", analysis_results['uncertainty']['ece'])


