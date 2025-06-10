import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import scipy.stats as stats

import os
import glob


class UncertaintyCalibrationAnalyser:
    def __init__(self, result_dir, file_pattern="*_uncertainty.json", num_bins=5, mode='all'):
        """
        Initialize the analyzer with multiple result files

        Args:
            result_dir (str): Path to directory containing JSON results
            file_pattern (str): Pattern to match result files
            num_bins (int): Number of bins for uncertainty segmentation
            mode: 'all' (default) uses all answers, 'original' uses only original answers
        """
        self.result_files = self._find_result_files(result_dir, file_pattern)
        self.num_bins = num_bins
        self.df = self._load_and_preprocess_data()
        self.bin_edges = np.linspace(0, 1, num_bins + 1)
        self.mode = mode

    def _find_result_files(self, result_dir, pattern):
        """Find all result files matching the pattern"""
        search_path = os.path.join(result_dir, pattern)
        files = sorted(glob.glob(search_path))
        if not files:
            raise ValueError(f"No files found matching pattern: {search_path}")
        return files

    def _load_and_preprocess_data(self):
        """Load and concatenate all result files"""
        dfs = []
        for file in self.result_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                dfs.append(pd.DataFrame(data))
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue

        if not dfs:
            raise ValueError("No valid data files could be loaded")

        return pd.concat(dfs, ignore_index=True)

    def _get_uncertainty_bin(self, uncertainty):
        """Assign uncertainty value to a bin"""
        return pd.cut([uncertainty], bins=self.bin_edges, labels=range(self.num_bins))[0]

    def analyze_calibration(self, unc_type: str = 'overall'):
        # Flatten the data
        flat_df = self._get_flattened_data()

        # 1. Confidence Calibration
        conf_df = flat_df[flat_df['confidence'].notna()]
        conf_bins = pd.cut(conf_df['confidence'], bins=self.num_bins)
        conf_stats = (
            conf_df.groupby(conf_bins)
            .agg(
                mean_confidence=('confidence', 'mean'),
                accuracy=('correct', 'mean'),
                count=('correct', 'size')
            )
            .reset_index()
            .dropna()
        )
        conf_ece = self._calculate_ece(conf_stats, 'mean_confidence', len(conf_df))
        flat_df = conf_df
        # 2. Uncertainty Calibration (using 1 - uncertainty as "certainty")
        flat_df['uncertainty'] = flat_df['uncertainty'].apply(lambda d: d.get(unc_type, None))
        flat_df['uncertainty'] = flat_df['uncertainty'].apply(lambda x: x[1] if isinstance(x, tuple) else x)

        uncert_bins = pd.cut(flat_df['uncertainty'], bins=self.num_bins)
        uncert_stats = (
            flat_df.groupby(uncert_bins)
            .agg(
                mean_uncertainty=('uncertainty', 'mean'),
                mean_certainty=('uncertainty', lambda x: 1 - x.mean()),
                accuracy=('correct', 'mean'),
                count=('correct', 'size')
            )
            .reset_index()
            .dropna()
        )
        uncert_ece = self._calculate_ece(uncert_stats, 'mean_certainty', len(flat_df))

        return {
            'confidence': {
                'bin_stats': conf_stats,
                'ece': conf_ece,
                'calibration_curve': calibration_curve(
                    conf_df['correct'].astype(int),
                    conf_df['confidence'],
                    n_bins=self.num_bins
                )
            },
            'uncertainty': {
                'bin_stats': uncert_stats,
                'ece': uncert_ece,
                'calibration_curve': calibration_curve(
                    flat_df['correct'].astype(int),
                    1 - flat_df['uncertainty'],
                    n_bins=self.num_bins
                )
            }
        }

    def _get_flattened_data(self):
        flat_data = []
        for _, row in self.df.iterrows():
            # Original answer
            if self.mode == 'original':
                flat_data.append({
                    'idx': row['idx'],
                    'uncertainty': row['uncertainty'],
                    'answer': row['original_answer'],
                    'confidence': row['original_confidence'],
                    'correct': row['original_correct'],
                    'perturbation': 'original',
                    'expected': row['expected_output']
                })

            # Perturbed answers (if needed)
            if self.mode == 'all':
                for perturb_type, answers in row['perturbed_answers'].items():
                    for ans in answers:
                        flat_data.append({
                            'idx': row['idx'],
                            'uncertainty': row['uncertainty'],
                            'answer': ans['answer'],
                            'confidence': ans['confidence'],
                            'correct': ans['correct'],
                            'perturbation': perturb_type,
                            'expected': row['expected_output']
                        })
        return pd.DataFrame(flat_data)

    def _calculate_ece(self, bin_stats, pred_col, total_samples):
        """Generic ECE calculation that works for both metrics"""
        return np.sum(
            np.abs(bin_stats['accuracy'] - bin_stats[pred_col]) *
            (bin_stats['count'] / total_samples)
        )

    def save_results(self, analysis_results, output_path):
        # NEEDS ADJUSTMENT!!!!
        """Save analysis results to JSON file"""
        results = {
            'metrics': {
                'ece': analysis_results['ece'],
                'brier': analysis_results['brier']
            },
            'bin_stats': analysis_results['bin_stats'].to_dict(orient='records'),
            'calibration_curve': {
                'prob_true': analysis_results['calibration_curve'][0].tolist(),
                'prob_pred': analysis_results['calibration_curve'][1].tolist()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def generate_uncertainty_comparison(self, save_dir=None):
        """Generate comparison plots for all uncertainty types"""
        flat_df = self._get_flattened_data()
        results = {}

        for uncert_type in ['overall', 'temp', 'trigger', 'rephrase']:
            # Create certainty or uncertainty column for plotting
            flat_df['current_uncert'] = flat_df['uncertainty'].apply(lambda d: d[uncert_type])

            self._plot_uncertainty_distribution(flat_df, uncert_type, save_dir)
            stats_df = self._run_uncertainty_stats(flat_df, uncert_type)
            results[uncert_type] = stats_df

        return results

    def _plot_uncertainty_distribution(self, df, uncert_type, save_dir):
        """Plot uncertainty distribution as boxplot by correctness"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x="perturbation",
            y="current_uncert",
            hue="correct",
            data=df,
            palette={True: "green", False: "red"}
        )
        plt.title(f"{uncert_type.capitalize()} Uncertainty Distribution")
        plt.ylabel("Uncertainty Score")

        if save_dir:
            plt.savefig(f"{save_dir}/{uncert_type}_distribution.png", bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def _run_uncertainty_stats(self, df, uncert_type):
        """Run t-tests comparing uncertainty values for correct vs incorrect predictions"""
        stats_results = []

        for perturb in df['perturbation'].unique():
            subset = df[df['perturbation'] == perturb]
            if len(subset) > 1:
                correct_values = subset[subset['correct']]['current_uncert']
                incorrect_values = subset[~subset['correct']]['current_uncert']

                if len(correct_values) > 0 and len(incorrect_values) > 0:
                    t_stat, p_value = stats.ttest_ind(
                        correct_values,
                        incorrect_values,
                        equal_var=False
                    )
                    stats_results.append({
                        'perturbation': perturb,
                        'mean_correct': correct_values.mean(),
                        'mean_incorrect': incorrect_values.mean(),
                        'p_value': p_value
                    })

        return pd.DataFrame(stats_results)

    def plot_comparison(self, analysis_results, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Confidence plot
        conf = analysis_results['confidence']
        ax1.plot(conf['calibration_curve'][1], conf['calibration_curve'][0], 's-')
        ax1.set_title(f'Confidence Calibration (ECE={conf["ece"]:.3f})')

        # Uncertainty plot (note: 1-uncertainty = "certainty")
        uncert = analysis_results['uncertainty']
        ax2.plot(uncert['bin_stats']['mean_certainty'],
                 uncert['bin_stats']['accuracy'], 'o-')
        ax2.set_title(f'Uncertainty Calibration (ECE={uncert["ece"]:.3f})')

        for ax in (ax1, ax2):
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Actual Accuracy')
            ax.grid(True)

        plt.show()

        plt.close()

    def analyze_all_uncertainties(self):
        """Analyze all uncertainty types (overall, temp, trigger, rephrase)"""
        flat_df = self._get_flattened_data()
        results = {}

        for uncert_type in ['overall', 'temp', 'trigger', 'rephrase']:
            # Convert uncertainty to certainty
            flat_df['certainty'] = flat_df['uncertainty'].apply(lambda d: 1 - d[uncert_type])

            # Create bins based on certainty
            bins = pd.cut(flat_df['certainty'], bins=self.num_bins)

            # Compute bin statistics
            bin_stats = (
                flat_df.groupby(bins)
                .agg(
                    mean_certainty=('certainty', 'mean'),
                    accuracy=('correct', 'mean'),
                    count=('correct', 'size')
                )
                .reset_index()
                .dropna()
            )

            # Compute Expected Calibration Error (ECE)
            ece = np.sum(
                np.abs(bin_stats['accuracy'] - bin_stats['mean_certainty']) *
                (bin_stats['count'] / len(flat_df))
            )

            # Compute calibration curve
            prob_true, prob_pred = calibration_curve(
                flat_df['correct'].astype(int),
                flat_df['certainty'],
                n_bins=self.num_bins
            )

            # Store results
            results[uncert_type] = {
                'bin_stats': bin_stats,
                'ece': ece,
                'calibration_curve': (prob_true, prob_pred)
            }

        return results

    def plot_uncertainty_calibration(self, analysis_results, save_dir=None):
        """Plot calibration curves for all uncertainty types"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, uncert_type in enumerate(['overall', 'temp', 'trigger', 'rephrase']):
            data = analysis_results[uncert_type]
            ax = axes[idx]

            # Plot calibration curve
            ax.plot(
                data['calibration_curve'][1],
                data['calibration_curve'][0],
                'o-',
                label=f'ECE={data["ece"]:.3f}'
            )
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f'{uncert_type.capitalize()} Uncertainty Calibration')
            ax.set_xlabel('Predicted Certainty (1 - Uncertainty)')
            ax.set_ylabel('Actual Accuracy')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/uncertainty_calibration_comparison.png", dpi=300)
        else:
            plt.show()
        plt.close()


