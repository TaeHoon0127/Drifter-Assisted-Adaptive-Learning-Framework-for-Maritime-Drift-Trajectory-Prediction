from collections import Counter
from datetime import datetime
from pathlib import Path
import random

import numpy as np
from geopy.distance import geodesic
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


random.seed(42)
np.random.seed(42)

from publication_utils import arguments, csv_files

args = arguments("ablation")
if args.original_data_path is None:
    raise ValueError("--original-data-path is required")
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIRS = [args.data_path]
RESULT_FILE = Path(args.save_path) / "expanded_svr_epsilon_ablation.txt"
TIMESTEP = 6
TEST_OFFSET = TIMESTEP - 1
N_FEATURES = 30
VALIDATION_EPSILONS = [0.01, 0.03, 0.1, 0.3, 1.0]

TASKS = {
    "prediction": {
        "target_slice": slice(30, 32),
        "add_model_var": False,
    },
    "correction": {
        "target_slice": slice(32, 34),
        "add_model_var": True,
    },
}

if args.task != "both":
    TASKS = {args.task: TASKS[args.task]}

EXPERIMENTS = [
    {
        "name": "raw_target_eps_0_1",
        "scale_target": False,
        "epsilon": 0.1,
        "select_epsilon": False,
    },
    {
        "name": "raw_target_eps_0_01",
        "scale_target": False,
        "epsilon": 0.01,
        "select_epsilon": False,
    },
    {
        "name": "standardized_target_eps_0_1",
        "scale_target": True,
        "epsilon": 0.1,
        "select_epsilon": False,
    },
    {
        "name": "standardized_target_validation_eps",
        "scale_target": True,
        "epsilon": None,
        "select_epsilon": True,
    },
]


def calculate_ncls(obs, pred):
    obs_loc = obs.copy()
    pred_loc = pred.copy()
    obs_loc[:, [0, 1]] = obs_loc[:, [1, 0]]
    pred_loc[:, [0, 1]] = pred_loc[:, [1, 0]]

    d = []
    for i in range(1, len(obs_loc)):
        d.append(geodesic(obs_loc[i], pred_loc[i]).km)
    d = np.array(d)

    dl = []
    for i in range(len(obs_loc) - 1):
        dl.append(geodesic(obs_loc[i], obs_loc[i + 1]).km)
    dl = np.array(dl)

    l = [0]
    for i in range(len(dl)):
        l.append(l[i] + dl[i])
    l = np.array(l)
    l = np.delete(l, 0, axis=0)

    denominator = np.sum(l)
    if denominator == 0:
        return np.nan
    s = np.sum(d) / denominator
    if s > 1:
        return 0
    return 1 - s


def loadtxt_2d(filename, **kwargs):
    return np.atleast_2d(np.loadtxt(filename, **kwargs))


def original_dataset_label(expanded_dataset_label):
    return args.original_data_path


def load_dataset(dataset_dir, target_slice):
    files = [dataset_dir / name for name in csv_files(dataset_dir, sort=True)]
    data_list = []
    target_list = []

    for filename in files:
        file_data = loadtxt_2d(filename, skiprows=1, delimiter=",")
        data_list.append(file_data[:, :N_FEATURES])
        target_list.append(file_data[:, target_slice])

    return files, data_list, target_list


def build_train_set(data_list, target_list, train_indices):
    data_train = np.concatenate([data_list[i] for i in train_indices], axis=0)
    target_train = np.concatenate([target_list[i] for i in train_indices], axis=0)
    return data_train, target_train


def load_observation(original_file):
    file_data = loadtxt_2d(
        original_file,
        skiprows=TIMESTEP + 3,
        delimiter=",",
        usecols=(8, 9, 10, 11),
    )
    start_point = np.loadtxt(
        original_file,
        skiprows=TIMESTEP + 2,
        max_rows=1,
        delimiter=",",
        usecols=(0, 1),
    )
    start_point = np.reshape(start_point, (1, 2))
    model_var = file_data[:, :2]
    obs_var = file_data[:, 2:]
    return start_point, model_var, obs_var


def make_location(start_point, displacement):
    cumulative = np.cumsum(displacement, axis=0)
    return np.vstack([start_point, start_point[0] + cumulative])


def make_svr(epsilon):
    return MultiOutputRegressor(
        SGDRegressor(
            loss="epsilon_insensitive",
            epsilon=epsilon,
            random_state=42,
        )
    )


def fit_predict_svr(data_train, target_train, data_test, epsilon, scale_target):
    x_scaler = StandardScaler()
    norm_data_train = x_scaler.fit_transform(data_train)
    norm_data_test = x_scaler.transform(data_test)

    if scale_target:
        y_scaler = StandardScaler()
        model_target_train = y_scaler.fit_transform(target_train)
    else:
        y_scaler = None
        model_target_train = target_train

    svr = make_svr(epsilon)
    svr.partial_fit(norm_data_train, model_target_train)
    pred_var = svr.predict(norm_data_test)

    if y_scaler is not None:
        pred_var = y_scaler.inverse_transform(pred_var)

    return pred_var


def evaluate(start_point, model_var, obs_var, pred_var, add_model_var):
    if add_model_var:
        eval_var = model_var + pred_var
    else:
        eval_var = pred_var

    mae = mean_absolute_error(obs_var, eval_var)
    obs_loc = make_location(start_point, obs_var)
    pred_loc = make_location(start_point, eval_var)
    ncls = calculate_ncls(obs_loc, pred_loc)
    return mae, ncls


def validate_epsilon(
    expanded_files,
    original_dir,
    data_list,
    target_list,
    train_indices,
    add_model_var,
):
    scores = {}

    for epsilon in VALIDATION_EPSILONS:
        fold_mae = []
        for val_idx in train_indices:
            inner_train_indices = [idx for idx in train_indices if idx != val_idx]
            if not inner_train_indices:
                continue

            data_train, target_train = build_train_set(
                data_list,
                target_list,
                inner_train_indices,
            )
            data_val = data_list[val_idx][TEST_OFFSET:, :]
            original_file = original_dir / expanded_files[val_idx].name
            start_point, model_var, obs_var = load_observation(original_file)

            if len(data_val) != len(obs_var):
                raise ValueError(
                    f"Length mismatch in validation file {expanded_files[val_idx]}: "
                    f"data={len(data_val)}, obs={len(obs_var)}"
                )

            pred_var = fit_predict_svr(
                data_train,
                target_train,
                data_val,
                epsilon,
                scale_target=True,
            )
            mae, _ = evaluate(start_point, model_var, obs_var, pred_var, add_model_var)
            fold_mae.append(mae)

        scores[epsilon] = float(np.mean(fold_mae)) if fold_mae else float("inf")

    selected = min(VALIDATION_EPSILONS, key=lambda eps: (scores[eps], eps))
    return selected, scores


def init_experiment_result():
    return {
        experiment["name"]: {
            "mae": [],
            "ncls": [],
            "epsilon": [],
            "folds": [],
        }
        for experiment in EXPERIMENTS
    }


def run_task_dataset(task_name, task_config, dataset_label):
    dataset_dir = (BASE_DIR / dataset_label).resolve()
    original_dir = (BASE_DIR / original_dataset_label(dataset_label)).resolve()
    if not dataset_dir.exists():
        return {
            "task": task_name,
            "dataset": dataset_label,
            "skipped": True,
            "reason": f"{dataset_dir} does not exist",
        }
    if not original_dir.exists():
        return {
            "task": task_name,
            "dataset": dataset_label,
            "skipped": True,
            "reason": f"{original_dir} does not exist",
        }

    files, data_list, target_list = load_dataset(
        dataset_dir,
        task_config["target_slice"],
    )
    if len(files) < 3:
        return {
            "task": task_name,
            "dataset": dataset_label,
            "skipped": True,
            "reason": "Need at least three files for nested validation",
        }

    result = {
        "task": task_name,
        "dataset": dataset_label,
        "dataset_path": str(dataset_dir),
        "original_dataset_path": str(original_dir),
        "skipped": False,
        "file_count": len(files),
        "baseline_mae": [],
        "baseline_ncls": [],
        "experiments": init_experiment_result(),
    }

    for test_idx, test_file in enumerate(files):
        train_indices = [idx for idx in range(len(files)) if idx != test_idx]
        data_train, target_train = build_train_set(data_list, target_list, train_indices)
        data_test = data_list[test_idx][TEST_OFFSET:, :]
        original_file = original_dir / test_file.name
        start_point, model_var, obs_var = load_observation(original_file)

        if len(data_test) != len(obs_var):
            raise ValueError(
                f"Length mismatch in test file {test_file}: "
                f"data={len(data_test)}, obs={len(obs_var)}"
            )

        baseline_mae, baseline_ncls = evaluate(
            start_point,
            model_var,
            obs_var,
            model_var,
            add_model_var=False,
        )
        result["baseline_mae"].append(baseline_mae)
        result["baseline_ncls"].append(baseline_ncls)

        for experiment in EXPERIMENTS:
            if experiment["select_epsilon"]:
                epsilon, validation_scores = validate_epsilon(
                    files,
                    original_dir,
                    data_list,
                    target_list,
                    train_indices,
                    task_config["add_model_var"],
                )
            else:
                epsilon = experiment["epsilon"]
                validation_scores = None

            pred_var = fit_predict_svr(
                data_train,
                target_train,
                data_test,
                epsilon,
                experiment["scale_target"],
            )
            mae, ncls = evaluate(
                start_point,
                model_var,
                obs_var,
                pred_var,
                task_config["add_model_var"],
            )

            exp_result = result["experiments"][experiment["name"]]
            exp_result["mae"].append(mae)
            exp_result["ncls"].append(ncls)
            exp_result["epsilon"].append(epsilon)
            exp_result["folds"].append(
                {
                    "file": test_file.name,
                    "epsilon": epsilon,
                    "mae": mae,
                    "ncls": ncls,
                    "validation_scores": validation_scores,
                }
            )

    return result


def mean_std(values):
    values = np.array(values, dtype=float)
    return float(np.mean(values)), float(np.std(values))


def format_scores(scores):
    if scores is None:
        return ""
    return ";".join(f"{epsilon}:{mae:.10f}" for epsilon, mae in scores.items())


def write_results(results):
    lines = [
        "Expanded SVR epsilon ablation",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Script: {Path(__file__).name}",
        f"Datasets: {', '.join(DATASET_DIRS)}",
        f"Validation epsilons: {VALIDATION_EPSILONS}",
        "Validation protocol: nested leave-one-file-out within each outer training set",
        "",
    ]

    for result in results:
        lines.append(f"Task: {result['task']}")
        lines.append(f"Dataset: {result['dataset']}")

        if result["skipped"]:
            lines.append(f"SKIP: {result['reason']}")
            lines.append("")
            continue

        lines.append(f"Dataset path: {result['dataset_path']}")
        lines.append(f"Original dataset path: {result['original_dataset_path']}")
        lines.append(f"Files: {result['file_count']}")

        baseline_mae_mean, baseline_mae_std = mean_std(result["baseline_mae"])
        baseline_ncls_mean, baseline_ncls_std = mean_std(result["baseline_ncls"])
        lines.append(
            "Baseline "
            f"MAE={baseline_mae_mean:.10f} +/- {baseline_mae_std:.10f}, "
            f"NCLS={baseline_ncls_mean:.10f} +/- {baseline_ncls_std:.10f}"
        )

        for experiment in EXPERIMENTS:
            exp_result = result["experiments"][experiment["name"]]
            mae_mean, mae_std = mean_std(exp_result["mae"])
            ncls_mean, ncls_std = mean_std(exp_result["ncls"])
            lines.append(
                f"{experiment['name']} "
                f"MAE={mae_mean:.10f} +/- {mae_std:.10f}, "
                f"NCLS={ncls_mean:.10f} +/- {ncls_std:.10f}"
            )
            if experiment["select_epsilon"]:
                selected_counts = Counter(exp_result["epsilon"])
                selected_text = ", ".join(
                    f"{epsilon}:{selected_counts[epsilon]}"
                    for epsilon in sorted(selected_counts)
                )
                lines.append(f"Selected epsilon counts: {selected_text}")

        lines.append("")
        lines.append("Per-fold results")
        lines.append("file,experiment,epsilon,mae,ncls,validation_mae_by_epsilon")
        for experiment in EXPERIMENTS:
            for fold in result["experiments"][experiment["name"]]["folds"]:
                lines.append(
                    f"{fold['file']},{experiment['name']},{fold['epsilon']},"
                    f"{fold['mae']:.10f},{fold['ncls']:.10f},"
                    f"{format_scores(fold['validation_scores'])}"
                )
        lines.append("")

    RESULT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    results = []
    for task_name, task_config in TASKS.items():
        for dataset_label in DATASET_DIRS:
            print(f"Running {task_name} / {dataset_label}")
            results.append(run_task_dataset(task_name, task_config, dataset_label))

    write_results(results)
    print(f"Saved results to {RESULT_FILE}")


if __name__ == "__main__":
    main()
