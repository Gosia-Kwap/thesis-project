import re
import math
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import scipy.stats as stats

from src import uncertainty
from src.utils.log_functions import log_message
from src.utils.Enums import LEVEL
from src.evaluate_uncertainty import extract_confidence

import os
import glob


class UncertaintyCalibrationAnalyser:
    def __init__(self, result_dir, file_pattern="*_uncertainty.json", num_bins=5, mode='original'):
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

    def analyze_calibration_conf_unc(self, unc_type: str = 'overall', save_dir=None):
        # Flatten the data
        flat_df = self._get_flattened_data()

        # 1. Confidence Calibration
        conf_df = flat_df[flat_df['confidence'].notna()]
        print(f"# unique confidence values: {conf_df['confidence'].nunique()}")
        try:
            conf_bins = pd.qcut(conf_df['confidence'], q=self.num_bins, duplicates='drop')
        except ValueError:
            # fallback: if too few unique values, use pd.cut
            log_message('Too few unique confidence values')
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

        try:
            uncert_bins = pd.qcut(conf_df['uncertainty'], q=self.num_bins, duplicates='drop')
        except ValueError:
            # fallback: if too few unique values, use pd.cut
            log_message('Too few uncertainty values')
            uncert_bins = pd.cut(conf_df['uncertainty'], bins=self.num_bins)
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

        results =  {
            'confidence': {
                'bin_stats': conf_stats,
                'ece': conf_ece,
                'calibration_curve': (conf_stats['accuracy'].to_numpy(), conf_stats['mean_confidence'].to_numpy())
            },
            'uncertainty': {
                'bin_stats': uncert_stats,
                'ece': uncert_ece,
                'calibration_curve': (uncert_stats['accuracy'].to_numpy(), uncert_stats['mean_certainty'].to_numpy())
            }
        }

        if save_dir:
            self.save_calibration_conf_unc(results, f"{save_dir}_calibration_unc_conf_results.json")

        return results

    def _get_flattened_data(self):
        flat_data = []
        for _, row in self.df.iterrows():
            # Original answer
            if self.mode == 'original':
                if math.isnan(row['original_confidence']):
                    row['original_confidence'] = extract_confidence(row['full_original_answer'])
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

        # Turn into DataFrame
        flat_df = pd.DataFrame(flat_data)

        # log NaN counts
        nan_conf_count = flat_df['confidence'].isna().sum()
        nan_answer_count = flat_df['answer'].isna().sum()

        log_message(f"Number of NaN confidence values: {nan_conf_count}")
        log_message(f"Number of NaN answer values: {nan_answer_count}")
        return flat_df

    def _calculate_ece(self, bin_stats, pred_col, total_samples):
        """Generic ECE calculation that works for both metrics"""
        return np.sum(
            np.abs(bin_stats['accuracy'] - bin_stats[pred_col]) *
            (bin_stats['count'] / total_samples)
        )

    def save_results(self, analysis_results, output_path):
        def convert(obj):
            if isinstance(obj, dict):
                return {convert(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="list")  # or "records" if you prefer row-wise dicts
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, tuple):
                return [convert(o) for o in obj]
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, pd.Interval):
                return str(obj)
            else:
                return obj

        cleaned = convert(analysis_results)

        with open(output_path, 'w') as f:
            json.dump(cleaned, f, indent=2)

    def generate_distribution_comparison_per_uncertainty(self, plot_dir=None, stats_dir: str = None):
        """Generate combined distribution comparison plot for all uncertainty types"""
        flat_df = self._get_flattened_data()
        results = {}

        # Prepare data for plotting - combine all uncertainty types
        plot_data = []
        for uncert_type in ['overall', 'temp', 'trigger', 'rephrase']:
            temp_df = flat_df.copy()
            temp_df['current_uncert'] = temp_df['uncertainty'].apply(lambda d: d[uncert_type])
            temp_df['uncertainty_type'] = uncert_type
            plot_data.append(temp_df)

        combined_df = pd.concat(plot_data)
        self._plot_uncertainty_distribution(combined_df, plot_dir)

        # Still generate individual stats for each type
        for uncert_type in ['overall', 'temp', 'trigger', 'rephrase']:
            stats_df = self._run_uncertainty_stats(
                combined_df[combined_df['uncertainty_type'] == uncert_type]
            )
            results[uncert_type] = stats_df

        if stats_dir:
            self.save_results(results, f"{stats_dir}_distribution_results.json")

        return results

    def _plot_uncertainty_distribution(self, combined_df, plot_dir):
        """Plot uncertainty distribution as boxplot by correctness"""
        # Create single comparison plot
        plt.figure(figsize=(16, 8))
        ax = sns.boxplot(
            x="uncertainty_type",
            y="current_uncert",
            hue="correct",
            data=combined_df,
            palette={True: "green", False: "red"},
            order=['overall', 'temp', 'trigger', 'rephrase']
        )

        plt.title("Uncertainty Distribution Comparison by Type")
        plt.xlabel("Uncertainty Type")
        plt.ylabel("Uncertainty Score")
        plt.legend(title="Correct", labels=["Correct", "Incorrect"])

        # Add statistical annotations if desired
        # (Could add significance markers here)

        if plot_dir:
            plt.savefig(f"{plot_dir}_combined_uncertainty_distributions.png",
                        bbox_inches='tight', dpi=300)
        else:
            plt.show()
        plt.close()

    def _run_uncertainty_stats(self, df):
        """Run t-tests comparing uncertainty values for correct vs incorrect predictions"""
        stats_results = []

        for perturb in df['perturbation'].unique():     # perturbation type is for if you analyse not only the original sample's distribution (resigned from it but just in case let's leave it here)
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
                        'p_value': p_value,
                        't_stat': t_stat,
                    })

        return pd.DataFrame(stats_results)

    def plot_calibration_unc_conf(self, analysis_results, save_path=None, mode='both'):
        """Plot calibration curves with mode options:
        - 'both': show confidence and uncertainty plots side-by-side (default)
        - 'confidence': show only confidence calibration
        - 'uncertainty': show only uncertainty calibration
        """
        # Create appropriate figure layout based on mode
        if mode == 'both':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax1 = ax if mode == 'confidence' else None
            ax2 = ax if mode == 'uncertainty' else None

        # Confidence plot
        if mode in ['both', 'confidence']:
            conf = analysis_results['confidence']
            conf_curve_y, conf_curve_x = conf['calibration_curve']
            conf_stats = conf['bin_stats']

            ax1.plot(conf_curve_x, conf_curve_y, 'o-', color='orange', label='Calibration curve')
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
            ax1.set_title(f'Confidence Calibration (ECE={conf["ece"]:.3f})')
            ax1.set_xlabel('Predicted Probability')
            ax1.set_ylabel('Actual Accuracy')
            ax1.grid(True)
            ax1.legend()

            # Bar chart showing counts per bin
            ax1b = ax1.twinx()
            bin_width = (conf_stats['mean_confidence'].max() - conf_stats['mean_confidence'].min()) / len(conf_stats)
            bar_width = bin_width * 0.8
            max_count = conf_stats['count'].max()
            conf_stats['count_scaled'] = conf_stats['count'] / (max_count * 4)

            ax1b.bar(conf_stats['mean_confidence'], conf_stats['count_scaled'],
                     width=bar_width, alpha=0.15, color='#6699cc', edgecolor='black', label='Count (scaled)')
            ax1b.set_ylim(0, 1)
            ax1b.set_ylabel('Count (scaled)', color='blue')
            ax1b.tick_params(axis='y', labelcolor='blue')

        # Uncertainty plot
        if mode in ['both', 'uncertainty']:
            uncert = analysis_results['uncertainty']
            uncert_curve_y, uncert_curve_x = uncert['calibration_curve']
            uncert_stats = uncert['bin_stats']

            ax2.plot(uncert_curve_x, uncert_curve_y, 'o-', color='orange', label='Calibration curve')
            ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
            ax2.set_title(f'Uncertainty Calibration (ECE={uncert["ece"]:.3f})')
            ax2.set_xlabel('Predicted Probability')
            ax2.set_ylabel('Actual Accuracy')
            ax2.grid(True)
            ax2.legend()

            # Bar chart showing counts per bin
            ax2b = ax2.twinx()
            bin_width = (uncert_stats['mean_certainty'].max() - uncert_stats['mean_certainty'].min()) / len(
                uncert_stats)
            bar_width = bin_width * 0.8
            max_count = uncert_stats['count'].max()
            uncert_stats['count_scaled'] = uncert_stats['count'] / (max_count * 4)

            ax2b.bar(uncert_stats['mean_certainty'], uncert_stats['count_scaled'],
                     width=bar_width, alpha=0.15, color='#6699cc', edgecolor='black', label='Count (scaled)')
            ax2b.set_ylim(0, 1)
            ax2b.set_ylabel('Count (scaled)', color='blue')
            ax2b.tick_params(axis='y', labelcolor='blue')

        fig.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_calibration_{mode}.png", bbox_inches='tight', dpi=300)
        else:
            plt.show()
        plt.close()

    def analyze_all_uncertainties(self, save_path=None):
        """Analyze all uncertainty types (overall, temp, trigger, rephrase)"""
        flat_df = self._get_flattened_data()
        results = {}

        for uncert_type in ['overall', 'temp', 'trigger', 'rephrase']:
            # Convert uncertainty to certainty
            flat_df['certainty'] = flat_df['uncertainty'].apply(lambda d: 1 - d[uncert_type])

            # Create bins based on certainty
            bins = pd.qcut(flat_df['certainty'], q=self.num_bins, duplicates='drop')

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
                n_bins=self.num_bins,
                strategy='quantile'
            )

            # Store results
            results[uncert_type] = {
                'bin_stats': bin_stats,
                'ece': ece,
                'calibration_curve': (prob_true, prob_pred)
            }
        #
        if save_path:
            savable_results = {}
            for uncert_type in ['overall', 'temp', 'trigger', 'rephrase']:
                savable_results[uncert_type] = {
                    'ece': results[uncert_type]['ece'],
                }
            self.save_results(savable_results, f"{save_path}_calibration_per_unc_results.json")

        return results

    def plot_calibration_all_unc(self, analysis_results, save_dir=None):
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
            plt.savefig(f"{save_dir}_uncertainty_calibration_comparison.png", dpi=300)
        else:
            plt.show()
        plt.close()

    def save_calibration_conf_unc(self, analysis_results, output_path):
        """Save full calibration analysis results to JSON file"""

        def convert_interval(obj):
            if isinstance(obj, pd.Interval):
                return str(obj)
            return obj

        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, pd.DataFrame):
                # Convert the entire DataFrame, handling Interval objects
                df = obj.copy()
                if isinstance(df.index, pd.IntervalIndex):
                    df.index = df.index.astype(str)
                # Convert all Interval objects in DataFrame columns
                df = df.applymap(convert_interval)
                return df.to_dict(orient='records')
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.float16, np.int32, np.int64)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, tuple):
                return [convert_for_json(o) for o in obj]
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            else:
                return convert_interval(obj)

        # Convert the full analysis results
        results_to_save = convert_for_json(analysis_results)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

