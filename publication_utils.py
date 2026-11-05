"""Small path/IO helpers and the agreed SVR correction for the original scripts."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("linear", "svr", "mlp", "lgbm", "lstm")


def arguments(kind="learning"):
    parser = argparse.ArgumentParser(description="Original experiment with configurable data_path/save_path.")
    parser.add_argument("--data-path", default="data_path")
    parser.add_argument("--save-path", default="save_path")
    parser.add_argument("--result-filename", help="Optional original log filename for the 24-hour wrappers")
    parser.add_argument("--original-data-path", help="Matching headerless hourly CSV directory")
    parser.add_argument("--pretrain-data-path", nargs=2, metavar=("SOURCE_1", "SOURCE_2"))
    parser.add_argument("--test-hours", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--incremental-epochs", type=int, default=300)
    parser.add_argument("--sort-files", action="store_true", help="Sort previously unsorted LOO/SHAP inputs")
    parser.add_argument("--file-order", help="Replay file_order.csv from an earlier LOO/SHAP run")
    if kind == "dataset":
        parser.add_argument("--directory-stride", type=int, default=1)
        parser.add_argument("--directory-offset", type=int, default=0)
        parser.add_argument("--invalid-rows", choices=("error", "skip"), default="error")
        parser.add_argument("--strict-continuity", action="store_true")
    if kind in ("shap", "svr", "ablation"):
        parser.add_argument("--dataset-label", default="dataset")
    if kind in ("svr", "ablation"):
        parser.add_argument("--task", choices=("prediction", "correction", "both"), default="both")
    if kind == "wma":
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--ga-weights-path")
        group.add_argument("--weights", nargs=5, type=float)
    if kind == "rbfn":
        parser.add_argument("--java-bin")
        parser.add_argument("--weka-classpath")
        parser.add_argument("--decimals", type=int, default=2)
        parser.add_argument("--distance", choices=("haversine", "geodesic"), default="haversine")
    args = parser.parse_args()
    if min(args.test_hours, args.epochs, args.incremental_epochs) < 1:
        parser.error("hours and epochs must be positive")
    for name in ("data_path", "save_path", "original_data_path", "file_order", "ga_weights_path"):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, str(Path(value).expanduser().resolve()))
    if args.pretrain_data_path:
        args.pretrain_data_path = [str(Path(p).expanduser().resolve()) for p in args.pretrain_data_path]
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    return args


def output_directory(args):
    """Keep original relative output names, contained in this experiment's save_path."""
    os.chdir(args.save_path)


def csv_files(directory, sort=False, order_file=None):
    names = [name for name in os.listdir(directory)
             if name.lower().endswith(".csv") and (Path(directory) / name).is_file()]
    if order_file:
        with open(order_file, newline="") as stream:
            ordered = [row["file_name"] for row in csv.DictReader(stream)]
        if len(ordered) != len(set(ordered)) or set(ordered) != set(names):
            raise ValueError("Saved file order does not match the input CSV cohort")
        return ordered
    return sorted(names) if sort else names


def save_file_order(path, names):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "file_name"))
        writer.writerows(enumerate(names))


class StandardizedTargetSVR:
    """Original partial_fit API; epsilon=0.1, frozen training-only target scaler."""
    def __init__(self):
        self.model = MultiOutputRegressor(SGDRegressor(
            loss="epsilon_insensitive", epsilon=0.1, random_state=42))
        self.target_scaler = StandardScaler()
        self.initialized = False

    def partial_fit(self, x, y):
        if not self.initialized:
            self.target_scaler.fit(y)
            self.initialized = True
        self.model.partial_fit(x, self.target_scaler.transform(y))
        return self

    def predict(self, x):
        return self.target_scaler.inverse_transform(self.model.predict(x))


def save_prediction_index(directory, pairs, hours, task, input_features):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pairs = list(pairs)
    indices, names = [p[0] for p in pairs], [p[1] for p in pairs]
    if len(set(indices)) != len(pairs) or len(set(names)) != len(pairs):
        raise ValueError("Duplicate prediction index or filename")
    with (directory / "prediction_index.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "file_name"))
        writer.writerows(pairs)
    (directory / "prediction_config.json").write_text(json.dumps({
        "test_hours": hours, "task": task, "input_features": list(input_features),
        "svr_epsilon": 0.1, "svr_standardized_target": True,
        "values": "final longitude/latitude displacement; correction already includes model displacement"
    }, indent=2) + "\n")


def prediction_pairs(directory, original_directory, hours):
    directory = Path(directory)
    index_file = directory / "prediction_index.csv"
    if not index_file.exists():
        raise ValueError(f"Missing {index_file}; regenerate predictions with the public scripts")
    with index_file.open(newline="") as stream:
        pairs = [(int(row["index"]), row["file_name"]) for row in csv.DictReader(stream)]
    if not pairs or len({p[0] for p in pairs}) != len(pairs) or len({p[1] for p in pairs}) != len(pairs):
        raise ValueError("Empty or duplicate prediction index")
    config = json.loads((directory / "prediction_config.json").read_text())
    if config["test_hours"] != hours:
        raise ValueError("Prediction horizon differs from requested evaluation horizon")
    for index, name in pairs:
        if Path(name).name != name or not (Path(original_directory) / name).is_file():
            raise ValueError(f"Invalid or missing original trajectory: {name}")
        for model in MODEL_NAMES:
            values = np.loadtxt(directory / f"{model}{index}.csv", delimiter=",", ndmin=2)
            if values.shape != (hours, 2) or not np.isfinite(values).all():
                raise ValueError(f"Invalid predictions for {model}, trajectory {name}")
    return pairs


def ga_and_test_pairs(test_dir, ga_dir, original_dir, hours, ga_hours):
    test = prediction_pairs(test_dir, original_dir, hours)
    objective = {name: index for index, name in prediction_pairs(ga_dir, original_dir, ga_hours)}
    if {name for _, name in test} != set(objective):
        raise ValueError("GA objective and test predictions must use the same trajectory cohort")
    a = json.loads((Path(test_dir) / "prediction_config.json").read_text())
    b = json.loads((Path(ga_dir) / "prediction_config.json").read_text())
    if a["task"] != b["task"] or a["input_features"] != b["input_features"]:
        raise ValueError("GA objective and test task/input features differ")
    if 2 * ga_hours > hours:
        raise ValueError("GA objective interval overlaps the evaluation interval")
    return [(name, index, objective[name]) for index, name in test]


def initial_weights(args):
    weights = args.weights if args.weights is not None else [1.0] * 5
    if args.ga_weights_path:
        with open(args.ga_weights_path, newline="") as stream:
            row = next(csv.DictReader(stream))
        weights = [float(row["weight_" + name]) for name in MODEL_NAMES]
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Initial weights must be finite, nonnegative, with a positive sum")
    return weights


def stable_weighted_majority_average(pred_stack, obs_var, weights):
    """Original predict-then-exp(-MAE) update, evaluated in log coordinates."""
    pred_stack = np.asarray(pred_stack)
    weights = np.asarray(weights, dtype=float)
    if pred_stack.shape != (len(weights), len(obs_var), 2):
        raise ValueError("Prediction and observation shapes differ")
    if not np.isfinite(pred_stack).all() or not np.isfinite(obs_var).all():
        raise ValueError("Non-finite WMA inputs")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid initial weights")
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    output = np.empty_like(obs_var, dtype=float)
    for j in range(len(obs_var)):
        weights = np.exp(log_weights - np.max(log_weights))
        output[j] = np.tensordot(weights / weights.sum(), pred_stack[:, j], axes=(0, 0))
        log_weights -= np.mean(np.abs(pred_stack[:, j] - obs_var[j]), axis=1)
        log_weights -= np.max(log_weights)
    return output
