"""Count each model's top-10 selections across seven 30-feature datasets."""
from collections import Counter
from pathlib import Path
import argparse
import csv
import json
import math
import re


ANALYSIS_NAMES = ("prediction", "correction")
MODELS = ("lgbm", "lstm")
N_FEATURES = 30
TOP_K = 10
OUTPUT_FILENAME = "shap_results"
FLOAT_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def validate_ranking(path, model, rows):
    if len(rows) != N_FEATURES:
        raise ValueError(f"{path}: {model} requires all 30 ranked features; found {len(rows)}")
    if {rank for rank, _, _ in rows} != set(range(1, N_FEATURES + 1)):
        raise ValueError(f"{path}: {model} has missing or duplicate ranks")
    if {feature for _, feature, _ in rows} != set(range(N_FEATURES)):
        raise ValueError(f"{path}: {model} must rank feature indices 0 through 29 exactly once")
    if any(not math.isfinite(score) or score < 0 for _, _, score in rows):
        raise ValueError(f"{path}: {model} has invalid SHAP scores")
    return [(feature, score) for rank, feature, score in sorted(rows) if rank <= TOP_K]


def read_csv_rankings(path, analysis_name):
    rankings = {model: [] for model in MODELS}
    labels = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            model = row["model"]
            if model not in rankings:
                raise ValueError(f"{path}: unexpected model {model}")
            if "task" in row and row["task"] != analysis_name:
                raise ValueError(f"{path}: task differs from {analysis_name}")
            if "n_features" in row and int(row["n_features"]) != N_FEATURES:
                raise ValueError(f"{path}: expected a 30-feature SHAP experiment")
            if "dataset_label" in row:
                labels.add(row["dataset_label"])
            rankings[model].append((int(row["rank"]), int(row["feature_index"]),
                                    float(row["shap_mean"])))
    if labels and (len(labels) != 1 or "" in labels):
        raise ValueError(f"{path}: expected one nonempty dataset_label")
    label = next(iter(labels)) if labels else path.stem.removeprefix(f"shap_{analysis_name}_")
    return label, {model: validate_ranking(path, model, rows)
                   for model, rows in rankings.items()}


def read_text_rankings(path, analysis_name):
    """Read original numpy arrays by model headings, independent of line wrapping."""
    text = path.read_text()
    headings = list(re.finditer(r"(?m)^(lgbm|lstm)[ \t]*$", text))
    rankings = {}
    for index, heading in enumerate(headings):
        model = heading.group(1)
        if model in rankings:
            raise ValueError(f"{path}: repeated {model} run; use one CSV snapshot per dataset")
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        arrays = re.findall(r"\[([^\[\]]*)\]", text[heading.end():end])
        if len(arrays) != 4:
            raise ValueError(f"{path}: expected full indices/scores and top-10 indices/scores")
        features = [int(value) for value in FLOAT_PATTERN.findall(arrays[0])]
        scores = [float(value) for value in FLOAT_PATTERN.findall(arrays[1])]
        top_features = [int(value) for value in FLOAT_PATTERN.findall(arrays[2])]
        top_scores = [float(value) for value in FLOAT_PATTERN.findall(arrays[3])]
        if len(features) != len(scores) or top_features != features[:TOP_K] or len(top_scores) != TOP_K:
            raise ValueError(f"{path}: inconsistent full and top-10 arrays")
        validate_ranking(path, model, [(rank, feature, score)
                                      for rank, (feature, score) in enumerate(zip(features, scores), 1)])
        if any(not math.isfinite(score) or score < 0 for score in top_scores):
            raise ValueError(f"{path}: invalid top-10 scores")
        rankings[model] = list(zip(top_features, top_scores))
    if set(rankings) != set(MODELS):
        raise ValueError(f"{path}: both lgbm and lstm results are required")
    label = path.stem.removeprefix(f"shap_{analysis_name}_").removeprefix("shap_")
    return label, rankings


def read_rankings(path, analysis_name):
    if path.suffix.lower() == ".csv":
        return read_csv_rankings(path, analysis_name)
    return read_text_rankings(path, analysis_name)


def get_input_paths(data_path, filenames):
    paths = [(Path(name) if Path(name).is_absolute() else data_path / name).resolve()
             for name in filenames]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate SHAP input paths")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SHAP inputs: " + ", ".join(missing))
    return paths


def summarize_feature_scores(input_paths, analysis_name, expected_datasets):
    if len(input_paths) != expected_datasets:
        raise ValueError(f"{analysis_name}: expected {expected_datasets} dataset files, found {len(input_paths)}")
    counts, shap_sums, dataset_counts = Counter(), Counter(), Counter()
    labels = set()
    for path in input_paths:
        label, rankings = read_rankings(path, analysis_name)
        if label in labels:
            raise ValueError(f"{analysis_name}: duplicate dataset_label {label}")
        labels.add(label)
        selected_in_dataset = set()
        for model in MODELS:
            for feature, score in rankings[model]:
                counts[feature] += 1
                shap_sums[feature] += score
                selected_in_dataset.add(feature)
        dataset_counts.update(selected_in_dataset)
    ranked_features = sorted(range(N_FEATURES),
                             key=lambda feature: (-counts[feature], -shap_sums[feature], feature))
    rows = [(feature, counts[feature], shap_sums[feature], dataset_counts[feature])
            for feature in ranked_features]
    return rows, sorted(labels)


def write_results(save_path, analysis_results):
    selections = {}
    with (save_path / OUTPUT_FILENAME).open("w") as output:
        output.write("# SHAP top-10 selection counts from 30 input features\n")
        output.write("# count: datasets x models; dataset_count: selected by either model\n")
        output.write("# sort: count desc, selected SHAP sum desc, feature_index asc\n\n")
        for analysis_name, rows, labels in analysis_results:
            output.write(f"[{analysis_name}]\n# datasets: {', '.join(labels)}\n")
            output.write(f"# source_files: {len(labels)}; source_results: {len(labels) * len(MODELS)}\n")
            output.write(f"# selected_features: {[row[0] for row in rows[:TOP_K]]}\n")
            output.write("feature_index\tcount\tshap_sum\tdataset_count\n")
            for feature, count, shap_sum, dataset_count in rows:
                output.write(f"{feature}\t{count}\t{shap_sum:.10f}\t{dataset_count}\n")
            output.write("\n")
            with (save_path / f"{analysis_name}_feature_counts.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("feature_index", "count", "shap_sum", "dataset_count"))
                writer.writerows(rows)
            selections[analysis_name] = {"selected_features": [row[0] for row in rows[:TOP_K]],
                                         "datasets": labels, "models": list(MODELS)}
    (save_path / "selected_features.json").write_text(json.dumps(selections, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data_path"))
    parser.add_argument("--save-path", type=Path, default=Path("save_path"))
    parser.add_argument("--prediction-files", nargs="+", default=[])
    parser.add_argument("--correction-files", nargs="+", default=[])
    parser.add_argument("--expected-datasets", type=int, default=7,
                        help="Expected datasets per task (default: 7; reduce for partial checks)")
    args = parser.parse_args()
    if args.expected_datasets < 1:
        parser.error("--expected-datasets must be positive")
    explicit_files = bool(args.prediction_files or args.correction_files)
    analysis_results = []
    for analysis_name in ANALYSIS_NAMES:
        filenames = getattr(args, analysis_name + "_files")
        if explicit_files:
            if not filenames:
                continue
            input_paths = get_input_paths(args.data_path, filenames)
        else:
            input_paths = sorted(path.resolve() for path in args.data_path.rglob(f"shap_{analysis_name}_*.csv"))
            if not input_paths:
                continue
        rows, labels = summarize_feature_scores(input_paths, analysis_name, args.expected_datasets)
        analysis_results.append((analysis_name, rows, labels))
    if not analysis_results:
        parser.error("No SHAP CSVs found; generate results or provide explicit input files")
    if len(analysis_results) == 2 and analysis_results[0][2] != analysis_results[1][2]:
        raise ValueError("Prediction and correction must use the same dataset labels")
    args.save_path.mkdir(parents=True, exist_ok=True)
    write_results(args.save_path, analysis_results)
    for analysis_name, rows, labels in analysis_results:
        print(f"{analysis_name}: {len(labels)} datasets, {len(labels) * len(MODELS)} model rankings; "
              f"selected features = {[row[0] for row in rows[:TOP_K]]}")


if __name__ == "__main__":
    main()
