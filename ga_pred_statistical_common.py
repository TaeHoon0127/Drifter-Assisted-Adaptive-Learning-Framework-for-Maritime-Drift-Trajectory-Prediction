import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from geopy.distance import geodesic
from sklearn.metrics import mean_absolute_error
from publication_utils import ga_and_test_pairs, stable_weighted_majority_average

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


MODEL_NAMES = ["linear", "svr", "mlp", "lgbm", "lstm"]
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.05
METHOD_OPENDRIFT = "OpenDrift"
METHOD_UNIFORM_AVERAGE = "Uniform_Average"
METHOD_GA_AVERAGE = "GA_Average"
METHOD_UNIFORM_WMA = "Uniform_WMA"
METHOD_GA_WMA = "GA_WMA"
METHOD_ORDER = [
    METHOD_OPENDRIFT,
    METHOD_UNIFORM_AVERAGE,
    METHOD_GA_AVERAGE,
    METHOD_UNIFORM_WMA,
    METHOD_GA_WMA,
]
GA_RUN_METHODS = [METHOD_GA_AVERAGE, METHOD_GA_WMA]
PLANNED_BASELINES = [
    METHOD_OPENDRIFT,
    METHOD_UNIFORM_AVERAGE,
    METHOD_GA_AVERAGE,
    METHOD_UNIFORM_WMA,
]

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILES = [
    "ga_run_summary.csv",
    "ga_run_fold_metrics.csv",
    "method_fold_metrics.csv",
    "summary_table.csv",
    "result_summary.csv",
    "ga_run_distribution_summary.csv",
    "wilcoxon_planned_comparisons.csv",
    "ga_selected_weights.csv",
    "manifest.json",
    "boxplots.png",
]


def resolve_path(path):
    return Path(path).expanduser().resolve()


def seed_tensorflow(seed):
    try:
        import tensorflow as tf
    except ImportError:
        return None
    tf.random.set_seed(seed)
    return tf.__version__


def load_csv_matrix(path):
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def load_original_slice(path, skiprows, max_rows):
    # NumPy usecols are zero-based: columns 8,9 are the OpenDrift/model
    # displacement baseline, and columns 10,11 are the observed displacement.
    arr = np.loadtxt(
        path,
        skiprows=skiprows,
        max_rows=max_rows,
        delimiter=",",
        usecols=(8, 9, 10, 11),
    )
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :2], arr[:, 2:]


def load_start_point(path, skiprows):
    start_point = np.loadtxt(
        path,
        skiprows=skiprows,
        max_rows=1,
        delimiter=",",
        usecols=(0, 1),
    )
    return np.reshape(start_point, (1, 2))


def build_location(start_point, var):
    loc = np.empty((len(var) + 1, 2), dtype=float)
    loc[0] = np.reshape(start_point, (2,))
    for idx in range(len(var)):
        loc[idx + 1] = loc[idx] + var[idx]
    return loc


def calculate_ncls(obs, pred):
    obs_loc = obs.copy()
    pred_loc = pred.copy()
    obs_loc[:, [0, 1]] = obs_loc[:, [1, 0]]
    pred_loc[:, [0, 1]] = pred_loc[:, [1, 0]]

    distance_errors = []
    for idx in range(1, len(obs_loc)):
        distance_errors.append(geodesic(obs_loc[idx], pred_loc[idx]).km)
    distance_errors = np.array(distance_errors)

    path_lengths = []
    for idx in range(len(obs_loc) - 1):
        path_lengths.append(geodesic(obs_loc[idx], obs_loc[idx + 1]).km)
    path_lengths = np.array(path_lengths)

    cumulative_lengths = [0]
    for path_length in path_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + path_length)
    cumulative_lengths = np.delete(np.array(cumulative_lengths), 0, axis=0)

    denominator = np.sum(cumulative_lengths)
    if denominator == 0:
        return np.nan

    score_loss = np.sum(distance_errors) / denominator
    if score_loss > 1:
        return 0.0
    return float(1 - score_loss)


def weighted_ensemble(pred_stack, weights):
    return np.tensordot(np.asarray(weights, dtype=float), pred_stack, axes=(0, 0))


def weighted_majority_average(pred_stack, obs_var, initial_weights):
    return stable_weighted_majority_average(pred_stack, obs_var, initial_weights)


def prepare_folds(add_original, test_dir, ga_test_dir, test_hours, ga_test_hours):
    additional = ga_and_test_pairs(test_dir, ga_test_dir, add_original, test_hours, ga_test_hours)
    folds = []

    for fold_idx, (filename, test_index, ga_index) in enumerate(additional):
        original_path = add_original / filename
        train_model_var, train_obs_var = load_original_slice(
            original_path,
            skiprows=1 + ga_test_hours,
            max_rows=ga_test_hours,
        )
        test_model_var, test_obs_var = load_original_slice(
            original_path,
            skiprows=1 + test_hours,
            max_rows=test_hours,
        )
        test_start_point = load_start_point(original_path, skiprows=test_hours)

        train_preds = []
        test_preds = []
        for model_name in MODEL_NAMES:
            train_preds.append(load_csv_matrix(ga_test_dir / f"{model_name}{ga_index}.csv"))
            test_preds.append(load_csv_matrix(test_dir / f"{model_name}{test_index}.csv"))

        folds.append(
            {
                "fold": fold_idx,
                "file_name": filename,
                "original_path": original_path,
                "train_model_var": train_model_var,
                "train_obs_var": train_obs_var,
                "train_preds": np.stack(train_preds, axis=0),
                "test_model_var": test_model_var,
                "test_obs_var": test_obs_var,
                "test_start_point": test_start_point,
                "test_obs_loc": build_location(test_start_point, test_obs_var),
                "test_preds": np.stack(test_preds, axis=0),
            }
        )

    return folds


def validate_folds(folds, test_hours, ga_test_hours):
    if not folds:
        raise ValueError("No fold files were found.")

    for fold in folds:
        if fold["train_obs_var"].shape[0] != ga_test_hours:
            raise ValueError(
                f"Fold {fold['fold']} train rows are {fold['train_obs_var'].shape[0]}, "
                f"expected {ga_test_hours}."
            )
        if fold["test_obs_var"].shape[0] != test_hours:
            raise ValueError(
                f"Fold {fold['fold']} test rows are {fold['test_obs_var'].shape[0]}, "
                f"expected {test_hours}."
            )
        if fold["train_preds"].shape[1:] != fold["train_obs_var"].shape:
            raise ValueError(f"Train prediction shape mismatch in fold {fold['fold']}.")
        if fold["test_preds"].shape[1:] != fold["test_obs_var"].shape:
            raise ValueError(f"Test prediction shape mismatch in fold {fold['fold']}.")


def cost(weights, train_folds):
    ensemble_mae = []
    for fold in train_folds:
        ensemble_var = weighted_ensemble(fold["train_preds"], weights)
        ensemble_mae.append(mean_absolute_error(fold["train_obs_var"], ensemble_var))
    return float(np.mean(ensemble_mae))


def fitness_calculator(weights_list, train_folds):
    cost_list = np.array([cost(weights, train_folds) for weights in weights_list])
    max_cost = float(np.max(cost_list))
    min_cost = float(np.min(cost_list))
    spread = max_cost - min_cost
    fitness_list = max_cost - cost_list + spread / 2

    if np.sum(fitness_list) <= 0:
        fitness_list = np.ones_like(cost_list)

    return fitness_list, cost_list


def selection(fitness_list, py_rng):
    total_fitness = float(np.sum(fitness_list))
    if total_fitness <= 0:
        return py_rng.randrange(0, len(fitness_list))

    point = py_rng.uniform(0, total_fitness)
    fitness_sum = 0
    for idx in range(len(fitness_list)):
        fitness_sum += fitness_list[idx]
        if point <= fitness_sum:
            return idx
    return len(fitness_list) - 1


def mutation(offspring, py_rng):
    offspring = list(offspring)
    max_index = int(np.argmax(offspring))
    index = max_index
    while max_index == index:
        index = py_rng.randrange(0, len(offspring))

    offspring[max_index], offspring[index] = offspring[index], offspring[max_index]
    return offspring


def crossover(parent1, parent2, py_rng):
    offspring = []
    for idx in range(len(parent1)):
        if py_rng.random() < CROSSOVER_PROB:
            offspring.append(parent1[idx])
        else:
            offspring.append(parent2[idx])

    if py_rng.random() < MUTATION_PROB:
        offspring = mutation(offspring, py_rng)

    return offspring


def repair(offspring_list):
    repaired = []
    for offspring in offspring_list:
        offspring = np.asarray(offspring, dtype=float)
        total = np.sum(offspring)
        if total == 0:
            offspring = np.ones_like(offspring) / len(offspring)
        else:
            offspring = offspring / total
        repaired.append(offspring)
    return np.asarray(repaired)


def replacement(weights_list, fitness_list, offspring_list, offspring_fitness_list):
    parents = list(zip(fitness_list, weights_list))
    offsprings = list(zip(offspring_fitness_list, offspring_list))
    combined = parents + offsprings
    combined.sort(key=lambda item: item[0], reverse=True)
    combined = combined[: len(weights_list)]
    final_fitness, final_weights = zip(*combined)
    return np.asarray(final_weights), np.asarray(final_fitness)


def run_ga(population, generation, repeat, seed, train_folds, legacy_replacement=False):
    best_weights = []
    run_summary_rows = []

    for repeat_idx in range(repeat):
        run_id = repeat_idx + 1
        run_seed = seed + repeat_idx
        py_rng = random.Random(run_seed)
        np_rng = np.random.default_rng(run_seed)
        seed_tensorflow(run_seed)

        print(f"GA repeat: {run_id}/{repeat}")
        weights_list = np_rng.dirichlet([1] * len(MODEL_NAMES), population)
        fitness_list, _ = fitness_calculator(weights_list, train_folds)

        for _ in range(generation):
            offspring_list = []
            for _ in range(population):
                p1 = selection(fitness_list, py_rng)
                p2 = selection(fitness_list, py_rng)
                if fitness_list[p2] > fitness_list[p1]:
                    p1, p2 = p2, p1
                offspring_list.append(crossover(weights_list[p1], weights_list[p2], py_rng))

            offspring_list = repair(offspring_list)
            offspring_fitness_list, _ = fitness_calculator(offspring_list, train_folds)
            if legacy_replacement:
                weights_list, fitness_list = replacement(
                    weights_list, fitness_list, offspring_list, offspring_fitness_list)
            else:
                # Compare both groups on one common objective, not separately scaled fitness.
                combined = np.vstack((weights_list, offspring_list))
                _, combined_costs = fitness_calculator(combined, train_folds)
                weights_list = combined[np.argsort(combined_costs, kind="stable")[:population]]
                fitness_list, _ = fitness_calculator(weights_list, train_folds)

        final_costs = np.array([cost(weights, train_folds) for weights in weights_list])
        best_idx = int(np.argmin(final_costs))
        run_best_weights = np.asarray(weights_list[best_idx], dtype=float)
        best_weights.append(run_best_weights)

        run_summary_rows.append(
            {
                "run": run_id,
                "seed": run_seed,
                "ga_objective_mae": float(final_costs[best_idx]),
                **{
                    f"weight_{model_name}": float(run_best_weights[idx])
                    for idx, model_name in enumerate(MODEL_NAMES)
                },
            }
        )

    return np.asarray(best_weights), run_summary_rows


def evaluate_var(fold, pred_var):
    mae = mean_absolute_error(fold["test_obs_var"], pred_var)
    pred_loc = build_location(fold["test_start_point"], pred_var)
    ncls = calculate_ncls(fold["test_obs_loc"], pred_loc)
    return float(mae), float(ncls)


def predict_method(fold, method, weights=None):
    if method == METHOD_OPENDRIFT:
        return fold["test_model_var"]
    if method == METHOD_UNIFORM_AVERAGE:
        return np.mean(fold["test_preds"], axis=0)
    if method == METHOD_GA_AVERAGE:
        return weighted_ensemble(fold["test_preds"], weights)
    if method == METHOD_UNIFORM_WMA:
        return weighted_majority_average(
            fold["test_preds"],
            fold["test_obs_var"],
            np.ones(len(MODEL_NAMES), dtype=float),
        )
    if method == METHOD_GA_WMA:
        return weighted_majority_average(
            fold["test_preds"],
            fold["test_obs_var"],
            weights,
        )
    raise ValueError(f"Unknown method: {method}")


def evaluate_ga_runs(folds, best_weights, run_summary_rows):
    rows = []
    for run_idx, weights in enumerate(best_weights, start=1):
        seed = run_summary_rows[run_idx - 1]["seed"]
        for fold in folds:
            for method in GA_RUN_METHODS:
                pred_var = predict_method(fold, method, weights=weights)
                mae, ncls = evaluate_var(fold, pred_var)
                rows.append(
                    {
                        "run": run_idx,
                        "seed": seed,
                        "method": method,
                        "fold": fold["fold"],
                        "file_name": fold["file_name"],
                        "mae": mae,
                        "ncls": ncls,
                    }
                )
    return rows


def add_run_cv_summary(run_summary_rows, ga_run_fold_rows):
    rows_by_run = {}
    for row in ga_run_fold_rows:
        rows_by_run.setdefault((row["run"], row["method"]), []).append(row)

    for row in run_summary_rows:
        fold_count = None
        for method in GA_RUN_METHODS:
            run_rows = rows_by_run[(row["run"], method)]
            mae_values = np.array([metric_row["mae"] for metric_row in run_rows], dtype=float)
            ncls_values = np.array([metric_row["ncls"] for metric_row in run_rows], dtype=float)
            fold_count = int(len(run_rows))
            prefix = method.lower()
            row[f"{prefix}_cv_mae_mean"] = float(np.nanmean(mae_values))
            row[f"{prefix}_cv_mae_std"] = (
                float(np.nanstd(mae_values, ddof=1)) if len(mae_values) > 1 else 0.0
            )
            row[f"{prefix}_cv_ncls_mean"] = float(np.nanmean(ncls_values))
            row[f"{prefix}_cv_ncls_std"] = (
                float(np.nanstd(ncls_values, ddof=1)) if len(ncls_values) > 1 else 0.0
            )
        row["fold_count"] = fold_count


def select_typical_solution(best_weights, run_summary_rows):
    objective_values = np.array([row["ga_objective_mae"] for row in run_summary_rows], dtype=float)
    median_objective = float(np.median(objective_values))
    candidate_order = sorted(
        range(len(objective_values)),
        key=lambda idx: (abs(objective_values[idx] - median_objective), run_summary_rows[idx]["run"]),
    )
    selected_idx = candidate_order[0]
    selected_row = run_summary_rows[selected_idx]
    selected_weights = np.asarray(best_weights[selected_idx], dtype=float)

    for row in run_summary_rows:
        row["is_typical_solution"] = row["run"] == selected_row["run"]

    return {
        "selection_criterion": "run closest to median GA objective MAE across independent GA runs",
        "used_by_methods": f"{METHOD_GA_AVERAGE};{METHOD_GA_WMA}",
        "median_ga_objective_mae": median_objective,
        "selected_run": selected_row["run"],
        "selected_seed": selected_row["seed"],
        "ga_objective_mae": selected_row["ga_objective_mae"],
        "ga_average_cv_mae_mean": selected_row["ga_average_cv_mae_mean"],
        "ga_average_cv_mae_std": selected_row["ga_average_cv_mae_std"],
        "ga_average_cv_ncls_mean": selected_row["ga_average_cv_ncls_mean"],
        "ga_average_cv_ncls_std": selected_row["ga_average_cv_ncls_std"],
        "ga_wma_cv_mae_mean": selected_row["ga_wma_cv_mae_mean"],
        "ga_wma_cv_mae_std": selected_row["ga_wma_cv_mae_std"],
        "ga_wma_cv_ncls_mean": selected_row["ga_wma_cv_ncls_mean"],
        "ga_wma_cv_ncls_std": selected_row["ga_wma_cv_ncls_std"],
        **{
            f"weight_{model_name}": float(selected_weights[idx])
            for idx, model_name in enumerate(MODEL_NAMES)
        },
    }, selected_weights


def evaluate_all_methods(folds, selected_weights):
    rows = []
    for fold in folds:
        for method in METHOD_ORDER:
            weights = selected_weights if method in [METHOD_GA_AVERAGE, METHOD_GA_WMA] else None
            pred_var = predict_method(fold, method, weights=weights)
            mae, ncls = evaluate_var(fold, pred_var)
            rows.append(
                {
                    "method": method,
                    "fold": fold["fold"],
                    "file_name": fold["file_name"],
                    "mae": mae,
                    "ncls": ncls,
                }
            )
    return rows


def summary_stats(values, bootstrap_rng=None, bootstrap_samples=10000):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    if len(values) == 1:
        ci_low = values[0]
        ci_high = values[0]
    elif bootstrap_rng is None or bootstrap_samples <= 0:
        ci_low = np.nan
        ci_high = np.nan
    else:
        sample_idx = bootstrap_rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
        sample_means = values[sample_idx].mean(axis=1)
        ci_low, ci_high = np.percentile(sample_means, [2.5, 97.5])

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def summary_table_rows(run_summary_rows, method_fold_rows, bootstrap_rng, bootstrap_samples):
    rows = []

    rows.append(
        {
            "scope": "ga_runs",
            "method": "GA",
            "metric": "ga_objective_mae",
            **summary_stats(
                [row["ga_objective_mae"] for row in run_summary_rows],
                bootstrap_rng=bootstrap_rng,
                bootstrap_samples=bootstrap_samples,
            ),
        }
    )

    for method in GA_RUN_METHODS:
        prefix = method.lower()
        for metric_name, field_name in [
            ("cv_mae_mean", f"{prefix}_cv_mae_mean"),
            ("cv_ncls_mean", f"{prefix}_cv_ncls_mean"),
        ]:
            rows.append(
                {
                    "scope": "ga_runs",
                    "method": method,
                    "metric": metric_name,
                    **summary_stats(
                        [row[field_name] for row in run_summary_rows],
                        bootstrap_rng=bootstrap_rng,
                        bootstrap_samples=bootstrap_samples,
                    ),
                }
            )

    for method in METHOD_ORDER:
        method_rows = [row for row in method_fold_rows if row["method"] == method]
        for metric in ["mae", "ncls"]:
            rows.append(
                {
                    "scope": "cv_folds",
                    "method": method,
                    "metric": metric,
                    **summary_stats(
                        [row[metric] for row in method_rows],
                        bootstrap_rng=bootstrap_rng,
                        bootstrap_samples=bootstrap_samples,
                    ),
                }
            )

    return rows


def result_summary_rows(summary_rows):
    rows = []
    by_method = {}
    for row in summary_rows:
        if row["scope"] != "cv_folds":
            continue
        by_method.setdefault(row["method"], {})[row["metric"]] = row

    for method in METHOD_ORDER:
        metric_rows = by_method.get(method)
        if metric_rows is None:
            continue
        output = {"method": method}
        for metric in ["mae", "ncls"]:
            stats = metric_rows.get(metric, {})
            for key in ["n", "mean", "std", "median", "q25", "q75", "ci95_low", "ci95_high"]:
                output[f"{metric}_{key}"] = stats.get(key, np.nan)
        rows.append(output)
    return rows


def ga_run_distribution_rows(summary_rows):
    rows = []
    for row in summary_rows:
        if row["scope"] == "ga_runs":
            rows.append(row)
    return rows


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    idx = 0
    while idx < len(values):
        next_idx = idx + 1
        while next_idx < len(values) and np.isclose(values[order[next_idx]], values[order[idx]]):
            next_idx += 1
        avg_rank = (idx + 1 + next_idx) / 2
        ranks[order[idx:next_idx]] = avg_rank
        idx = next_idx
    return ranks


def wilcoxon_normal_approx(paired_delta):
    non_zero = paired_delta[~np.isclose(paired_delta, 0)]
    n = len(non_zero)
    if n == 0:
        return {
            "p_value": 1.0,
            "statistic": 0.0,
            "test": "wilcoxon_signed_rank_all_zero_differences",
        }

    abs_delta = np.abs(non_zero)
    ranks = average_ranks(abs_delta)
    w_plus = float(np.sum(ranks[non_zero > 0]))

    _, counts = np.unique(abs_delta, return_counts=True)
    tie_correction = np.sum(counts**3 - counts) / 48
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_correction
    if variance <= 0:
        return {
            "p_value": 1.0,
            "statistic": w_plus,
            "test": "wilcoxon_signed_rank_normal_approx_zero_variance",
        }

    z = (w_plus - mean - 0.5) / math.sqrt(variance)
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return {
        "p_value": float(p_value),
        "statistic": w_plus,
        "test": "wilcoxon_signed_rank_normal_approx_no_scipy",
    }


def wilcoxon_test(primary_values, baseline_values, metric):
    primary_values = np.asarray(primary_values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    valid = ~np.isnan(primary_values) & ~np.isnan(baseline_values)
    primary_values = primary_values[valid]
    baseline_values = baseline_values[valid]

    if metric == "mae":
        paired_delta = baseline_values - primary_values
        better_direction = "primary_lower"
    else:
        paired_delta = primary_values - baseline_values
        better_direction = "primary_higher"

    result = {
        "test": "wilcoxon_signed_rank",
        "n_pairs": int(len(paired_delta)),
        "alternative": "greater",
        "better_direction": better_direction,
        "mean_delta": float(np.mean(paired_delta)) if len(paired_delta) else np.nan,
        "median_delta": float(np.median(paired_delta)) if len(paired_delta) else np.nan,
        "statistic": np.nan,
        "p_value": np.nan,
    }

    if len(paired_delta) == 0 or np.allclose(paired_delta, 0):
        result["test"] = "wilcoxon_signed_rank_all_zero_differences"
        result["statistic"] = 0.0
        result["p_value"] = 1.0
        return result

    if wilcoxon is None:
        result.update(wilcoxon_normal_approx(paired_delta))
        return result

    try:
        stat = wilcoxon(paired_delta, alternative="greater", zero_method="wilcox")
        result["statistic"] = float(stat.statistic)
        result["p_value"] = float(stat.pvalue)
    except ValueError as exc:
        result["test"] = f"wilcoxon_signed_rank_unavailable_{exc}"

    return result


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    valid_idx = np.where(~np.isnan(p_values))[0]
    if len(valid_idx) == 0:
        return adjusted

    valid_p = p_values[valid_idx]
    order = np.argsort(valid_p)
    sorted_idx = valid_idx[order]
    sorted_p = valid_p[order]

    running_max = 0
    total = len(sorted_p)
    for rank, original_idx in enumerate(sorted_idx):
        adjusted_p = (total - rank) * sorted_p[rank]
        running_max = max(running_max, adjusted_p)
        adjusted[original_idx] = min(running_max, 1.0)
    return adjusted


def wilcoxon_rows(method_fold_rows):
    rows = []
    primary_rows = [row for row in method_fold_rows if row["method"] == METHOD_GA_WMA]
    primary_by_fold = {row["fold"]: row for row in primary_rows}

    for baseline_method in PLANNED_BASELINES:
        baseline_rows = [row for row in method_fold_rows if row["method"] == baseline_method]
        baseline_by_fold = {row["fold"]: row for row in baseline_rows}
        common_folds = sorted(set(primary_by_fold) & set(baseline_by_fold))

        for metric in ["mae", "ncls"]:
            primary_values = [primary_by_fold[fold][metric] for fold in common_folds]
            baseline_values = [baseline_by_fold[fold][metric] for fold in common_folds]
            rows.append(
                {
                    "primary_method": METHOD_GA_WMA,
                    "baseline_method": baseline_method,
                    "metric": metric,
                    **wilcoxon_test(primary_values, baseline_values, metric),
                }
            )

    adjusted = holm_adjust([row["p_value"] for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["p_value_holm"] = float(adjusted_p) if not np.isnan(adjusted_p) else np.nan
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(args):
    try:
        completed = subprocess.run(
            args,
            cwd=SCRIPT_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def make_boxplots(output_dir, run_summary_rows, method_fold_rows):
    if plt is None:
        return None

    run_mae = [
        [row[f"{method.lower()}_cv_mae_mean"] for row in run_summary_rows]
        for method in GA_RUN_METHODS
    ]
    run_ncls = [
        [row[f"{method.lower()}_cv_ncls_mean"] for row in run_summary_rows]
        for method in GA_RUN_METHODS
    ]
    fold_mae = [
        [row["mae"] for row in method_fold_rows if row["method"] == method]
        for method in METHOD_ORDER
    ]
    fold_ncls = [
        [row["ncls"] for row in method_fold_rows if row["method"] == method]
        for method in METHOD_ORDER
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].boxplot(run_mae, labels=GA_RUN_METHODS)
    axes[0, 0].set_title("Run-Level MAE")
    axes[0, 0].set_ylabel("Fold-Mean MAE")

    axes[0, 1].boxplot(run_ncls, labels=GA_RUN_METHODS)
    axes[0, 1].set_title("Run-Level NCLS")
    axes[0, 1].set_ylabel("Fold-Mean NCLS")

    axes[1, 0].boxplot(fold_mae, labels=METHOD_ORDER)
    axes[1, 0].set_title("Fold-Level MAE")
    axes[1, 0].set_ylabel("MAE")

    axes[1, 1].boxplot(fold_ncls, labels=METHOD_ORDER)
    axes[1, 1].set_title("Fold-Level NCLS")
    axes[1, 1].set_ylabel("NCLS")

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)

    fig.tight_layout()
    output_path = output_dir / "boxplots.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return str(output_path)


def build_manifest(
    args,
    script_path,
    add_original,
    test_dir,
    ga_test_dir,
    output_dir,
    folds,
    selected_solution,
    tensorflow_version,
    boxplot_path,
):
    script_path = Path(script_path).resolve()
    common_path = Path(__file__).resolve()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "common_module_path": str(common_path),
        "common_module_sha256": sha256_file(common_path),
        "publication_utils_sha256": sha256_file(SCRIPT_DIR / "publication_utils.py"),
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_value(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_value(["git", "status", "--short"]),
        "package_versions": {
            "numpy": np.__version__,
            "scikit-learn": package_version("scikit-learn"),
            "geopy": package_version("geopy"),
            "scipy": package_version("scipy"),
            "tensorflow": tensorflow_version,
            "matplotlib": package_version("matplotlib"),
        },
        "paths": {
            "original_data_dir": str(add_original),
            "test_prediction_dir": str(test_dir),
            "ga_objective_prediction_dir": str(ga_test_dir),
            "output_dir": str(output_dir),
        },
        "data_columns": {
            "indexing": "NumPy zero-based usecols",
            "usecols_8_9": "OpenDrift/model displacement variables, used as the paper baseline",
            "usecols_10_11": "observed displacement variables",
            "trajectory_reconstruction": "hourly displacement variables are cumulatively added from the start point",
        },
        "methods": {
            METHOD_OPENDRIFT: "OpenDrift/model displacement from data columns 8 and 9",
            METHOD_UNIFORM_AVERAGE: "fixed uniform average of the five model displacement predictions over all 48 hours",
            METHOD_GA_AVERAGE: "fixed average using the typical GA initial weights over all 48 hours",
            METHOD_UNIFORM_WMA: "WMA initialized uniformly, then updated at each hour",
            METHOD_GA_WMA: "WMA initialized with the typical GA weights, then updated at each hour",
        },
        "hyperparameters": {
            "population": args.population,
            "generation": args.generation,
            "repeat": args.repeat,
            "seed": args.seed,
            "per_run_seed": "seed + run_index_zero_based",
            "test_hours": args.test_hours,
            "ga_test_hours": args.ga_test_hours,
            "crossover_prob": CROSSOVER_PROB,
            "mutation_prob": MUTATION_PROB,
            "initial_population": "Dirichlet(alpha=[1, 1, 1, 1, 1])",
            "selection": "roulette wheel over GA fitness",
            "repair": "normalize weights to sum to 1",
            "replacement": "legacy relative fitness" if args.legacy_replacement else "combined population ranked by common MAE",
            "typical_solution": selected_solution["selection_criterion"],
            "wma_update": "weights_t+1 = weights_t * exp(-per-model absolute prediction error at t)",
            "bootstrap_samples": args.bootstrap,
            "statistical_test": "planned paired Wilcoxon signed-rank tests; primary method is GA_WMA",
            "planned_baselines": PLANNED_BASELINES,
            "multiple_testing_correction": "Holm over planned comparisons and metrics",
            "models": MODEL_NAMES,
        },
        "fold_count": len(folds),
        "fold_files": [fold["file_name"] for fold in folds],
        "ga_selected_weights": selected_solution,
        "outputs": {
            "csv_or_json": [name for name in OUTPUT_FILES if name != "boxplots.png"],
            "boxplot": boxplot_path,
        },
    }


def build_parser(default_test_dir, default_add_original, default_output_dir):
    parser = argparse.ArgumentParser(
        description=(
            "Run 30 GA optimizations, evaluate OpenDrift and four ensemble "
            "methods, deploy a typical median-objective GA solution, and save "
            "only reviewer-response statistics."
        )
    )
    parser.add_argument("--data-path", "--test-dir", dest="test_dir", default=default_test_dir)
    parser.add_argument("--ga-data-path", required=True)
    parser.add_argument("--original-data-path", "--add-original", dest="add_original", default=default_add_original)
    parser.add_argument("--save-path", "--output-dir", dest="output_dir", default=default_output_dir)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generation", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--test-hours", type=int, default=48)
    parser.add_argument("--ga-test-hours", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--skip-boxplot", action="store_true")
    parser.add_argument("--legacy-replacement", action="store_true", help="Historical relative-fitness comparison for result audits")
    return parser


def run(default_test_dir, default_add_original, default_output_dir, script_path):
    args = build_parser(default_test_dir, default_add_original, default_output_dir).parse_args()

    if min(args.population, args.repeat, args.test_hours, args.ga_test_hours) < 1 or args.generation < 0 or args.bootstrap < 0:
        raise ValueError("Invalid experiment counts")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tensorflow_version = seed_tensorflow(args.seed)

    add_original = resolve_path(args.add_original)
    test_dir = resolve_path(args.test_dir)
    ga_test_dir = resolve_path(args.ga_data_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = prepare_folds(
        add_original=add_original,
        test_dir=test_dir,
        ga_test_dir=ga_test_dir,
        test_hours=args.test_hours,
        ga_test_hours=args.ga_test_hours,
    )
    validate_folds(folds, args.test_hours, args.ga_test_hours)

    best_weights, run_summary_rows = run_ga(
        population=args.population,
        generation=args.generation,
        repeat=args.repeat,
        seed=args.seed,
        train_folds=folds,
        legacy_replacement=args.legacy_replacement,
    )
    ga_run_fold_rows = evaluate_ga_runs(folds, best_weights, run_summary_rows)
    add_run_cv_summary(run_summary_rows, ga_run_fold_rows)
    selected_solution, selected_weights = select_typical_solution(best_weights, run_summary_rows)
    method_fold_rows = evaluate_all_methods(folds, selected_weights)

    bootstrap_rng = np.random.default_rng(args.seed + 100000)
    summary_rows = summary_table_rows(
        run_summary_rows,
        method_fold_rows,
        bootstrap_rng=bootstrap_rng,
        bootstrap_samples=args.bootstrap,
    )
    result_rows = result_summary_rows(summary_rows)
    ga_distribution_rows = ga_run_distribution_rows(summary_rows)
    test_rows = wilcoxon_rows(method_fold_rows)

    boxplot_path = None
    if not args.skip_boxplot:
        boxplot_path = make_boxplots(output_dir, run_summary_rows, method_fold_rows)

    weight_fields = [f"weight_{model_name}" for model_name in MODEL_NAMES]
    write_csv(
        output_dir / "ga_run_summary.csv",
        run_summary_rows,
        [
            "run",
            "seed",
            "ga_objective_mae",
            "fold_count",
            "ga_average_cv_mae_mean",
            "ga_average_cv_mae_std",
            "ga_average_cv_ncls_mean",
            "ga_average_cv_ncls_std",
            "ga_wma_cv_mae_mean",
            "ga_wma_cv_mae_std",
            "ga_wma_cv_ncls_mean",
            "ga_wma_cv_ncls_std",
            "is_typical_solution",
            *weight_fields,
        ],
    )
    write_csv(
        output_dir / "ga_run_fold_metrics.csv",
        ga_run_fold_rows,
        ["run", "seed", "method", "fold", "file_name", "mae", "ncls"],
    )
    write_csv(
        output_dir / "method_fold_metrics.csv",
        method_fold_rows,
        ["method", "fold", "file_name", "mae", "ncls"],
    )
    write_csv(
        output_dir / "summary_table.csv",
        summary_rows,
        ["scope", "method", "metric", "n", "mean", "std", "median", "q25", "q75", "ci95_low", "ci95_high"],
    )
    write_csv(
        output_dir / "result_summary.csv",
        result_rows,
        [
            "method",
            "mae_n",
            "mae_mean",
            "mae_std",
            "mae_median",
            "mae_q25",
            "mae_q75",
            "mae_ci95_low",
            "mae_ci95_high",
            "ncls_n",
            "ncls_mean",
            "ncls_std",
            "ncls_median",
            "ncls_q25",
            "ncls_q75",
            "ncls_ci95_low",
            "ncls_ci95_high",
        ],
    )
    write_csv(
        output_dir / "ga_run_distribution_summary.csv",
        ga_distribution_rows,
        ["scope", "method", "metric", "n", "mean", "std", "median", "q25", "q75", "ci95_low", "ci95_high"],
    )
    write_csv(
        output_dir / "wilcoxon_planned_comparisons.csv",
        test_rows,
        [
            "primary_method",
            "baseline_method",
            "metric",
            "test",
            "n_pairs",
            "alternative",
            "better_direction",
            "mean_delta",
            "median_delta",
            "statistic",
            "p_value",
            "p_value_holm",
        ],
    )
    write_csv(
        output_dir / "ga_selected_weights.csv",
        [selected_solution],
        [
            "selection_criterion",
            "used_by_methods",
            "median_ga_objective_mae",
            "selected_run",
            "selected_seed",
            "ga_objective_mae",
            "ga_average_cv_mae_mean",
            "ga_average_cv_mae_std",
            "ga_average_cv_ncls_mean",
            "ga_average_cv_ncls_std",
            "ga_wma_cv_mae_mean",
            "ga_wma_cv_mae_std",
            "ga_wma_cv_ncls_mean",
            "ga_wma_cv_ncls_std",
            *weight_fields,
        ],
    )

    manifest = build_manifest(
        args=args,
        script_path=script_path,
        add_original=add_original,
        test_dir=test_dir,
        ga_test_dir=ga_test_dir,
        output_dir=output_dir,
        folds=folds,
        selected_solution=selected_solution,
        tensorflow_version=tensorflow_version,
        boxplot_path=boxplot_path,
    )
    write_json(output_dir / "manifest.json", manifest)

    print(f"Saved reviewer-response GA statistics to: {output_dir}")
    if boxplot_path is None and not args.skip_boxplot:
        print("Matplotlib is unavailable; boxplots.png was not created.")
