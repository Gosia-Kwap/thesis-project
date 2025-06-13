# from src.uncertainty.UncertaintyCalibrationAnalyser import UncertaintyCalibrationAnalyser
#
# result_dir = "/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/results-copy"
#
# analyzer_all = UncertaintyCalibrationAnalyser(
#     result_dir=result_dir,
#     file_pattern="SVAMP_perturbed_outputs_gemma9b_*_uncertainty.json",
#     num_bins=25,
#     mode='all'
# )
# results_all = analyzer_all.analyze_calibration()
#
# # Analyze only original answers
# analyzer_orig = UncertaintyCalibrationAnalyser(
#     result_dir=result_dir,
#     file_pattern="SVAMP_perturbed_outputs_gemma9b_*_uncertainty.json",
#     num_bins=25,
#     mode='original'
# )
# results_orig = analyzer_orig.analyze_calibration()
#
# # Compare results
# print("All answers confidence ECE:", results_all['confidence']['ece'])
# print("All answers uncertainty ECE:", results_all['uncertainty']['ece'])
# print("Original-only confidence ECE:", results_orig['confidence']['ece'])
# print("Original-only uncertainty ECE:", results_orig['uncertainty']['ece'])
#
#

from src.uncertainty.UncertaintyCalibrationAnalyser import UncertaintyCalibrationAnalyser

task = "SVAMP"
model = 'llama3'
method = 'cosine'
result_dir = "/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/uncertainty"
plot_dir = f"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/plots/{task}/"


# Initialize the analyser with the directory containing results
analyser = UncertaintyCalibrationAnalyser(
    result_dir=result_dir,
    file_pattern=f"{task}_perturbed_outputs_gemma9b_*_uncertainty_{method}.json",
    num_bins=6,
    mode="original"  # Use "original" to only analyze the original answers
)

# Analyze calibration (confidence and overall uncertainty)
analysis_results = analyser.analyze_calibration()

# Plot calibration comparison
analyser.plot_comparison(analysis_results)

# Optional: Save results to JSON
# analyser.save_results(analysis_results['confidence'], 'confidence_results.json')
# analyser.save_results(analysis_results['uncertainty'], 'uncertainty_results.json')

# Compare uncertainty distributions across types and perturbations
uncertainty_distributions = analyser.generate_uncertainty_comparison(save_dir=plot_dir, model=model)

# Perform separate calibration analysis per uncertainty type (overall, temp, trigger, rephrase)
all_uncertainty_results = analyser.analyze_all_uncertainties()

# Plot calibration curves for each uncertainty type
analyser.plot_uncertainty_calibration(all_uncertainty_results, save_dir=plot_dir)




