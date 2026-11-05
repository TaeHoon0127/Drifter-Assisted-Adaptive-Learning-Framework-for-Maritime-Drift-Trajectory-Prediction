from copy import deepcopy
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

from publication_utils import arguments, csv_files, save_prediction_index

args = arguments("svr")
if not args.pretrain_data_path or args.original_data_path is None:
    raise ValueError("--pretrain-data-path and --original-data-path are required")
BASE_DIR = Path(__file__).resolve().parent
RESULT_FILE = Path(args.save_path) / (args.result_filename or "expanded_incremental_svr_epsilon_ablation.txt")

DRIFTERS = ["source1", "source2", "target"]
DATA_PATHS = dict(zip(DRIFTERS, args.pretrain_data_path + [args.data_path]))
TEST_HOURS = args.test_hours
INCREMENTAL_END = TEST_HOURS - 3
TEST_START = TEST_HOURS - 3
TEST_END = TEST_HOURS * 2 - 3
N_FEATURES = 10
SAVE_INCREMENTAL_PREDICTIONS = True

TASKS = {
    "prediction": {
        "selected_features": [5, 8, 9, 12, 13, 20, 21, 26, 28, 29, 30, 31],
        "add_model_var": False,
        "dir_suffix": "pred",
    },
    "correction": {
        "selected_features": [4, 5, 10, 11, 12, 13, 19, 21, 28, 29, 32, 33],
        "add_model_var": True,
        "dir_suffix": "cor",
    },
}

if args.task != "both":
    TASKS = {args.task: TASKS[args.task]}

EXPERIMENTS = [
    {
        "name": "standardized_target_eps_0_1",
        "scale_target": True,
        "epsilon": 0.1,
    },
]

MODES = ["without incremental learning", "incremental learning"]


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


def expansion_dir(drifter):
    return Path(DATA_PATHS[drifter])


def original_dir(drifter):
    return Path(args.original_data_path)


def prediction_dir(add_drifter, task_config):
    return Path(args.save_path) / task_config["dir_suffix"] / "predictions"


def load_feature_file(filename, selected_features):
    file_data = loadtxt_2d(
        filename,
        skiprows=1,
        delimiter=",",
        usecols=selected_features,
    )
    return file_data[:, :N_FEATURES], file_data[:, N_FEATURES:N_FEATURES + 2]


def load_records(drifters, task_config):
    records = []
    for drifter in drifters:
        data_dir = expansion_dir(drifter)
        obs_dir = original_dir(drifter)
        if not data_dir.exists():
            raise FileNotFoundError(f"{data_dir} does not exist")
        if drifter == "target" and not obs_dir.exists():
            raise FileNotFoundError(f"{obs_dir} does not exist")

        files = [data_dir / name for name in csv_files(data_dir, sort=True)]
        for filename in files:
            obs_file = obs_dir / filename.name
            if drifter == "target" and not obs_file.exists():
                raise FileNotFoundError(f"{obs_file} does not exist")

            data, target = load_feature_file(filename, task_config["selected_features"])
            records.append(
                {
                    "drifter": drifter,
                    "file": filename,
                    "original_file": obs_file,
                    "data": data,
                    "target": target,
                }
            )

    return records


def concatenate(records, data_slice=None):
    data_parts = []
    target_parts = []
    for record in records:
        if data_slice is None:
            data_parts.append(record["data"])
            target_parts.append(record["target"])
        else:
            data_parts.append(record["data"][data_slice, :])
            target_parts.append(record["target"][data_slice, :])

    if not data_parts:
        raise ValueError("No records available for training")

    return np.concatenate(data_parts, axis=0), np.concatenate(target_parts, axis=0)


def load_observation(original_file):
    file_data = loadtxt_2d(
        original_file,
        skiprows=1 + TEST_HOURS,
        max_rows=TEST_HOURS,
        delimiter=",",
        usecols=(8, 9, 10, 11),
    )
    start_point = np.loadtxt(
        original_file,
        skiprows=TEST_HOURS,
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


def fit_svr(data_train, target_train, epsilon, scale_target):
    x_scaler = StandardScaler()
    norm_data_train = x_scaler.fit_transform(data_train)

    if scale_target:
        y_scaler = StandardScaler()
        model_target_train = y_scaler.fit_transform(target_train)
    else:
        y_scaler = None
        model_target_train = target_train

    model = make_svr(epsilon)
    model.partial_fit(norm_data_train, model_target_train)
    return model, x_scaler, y_scaler


def incremental_fit(model, x_scaler, y_scaler, data_train, target_train):
    incremental_model = deepcopy(model)
    norm_data_train = x_scaler.transform(data_train)

    if y_scaler is None:
        model_target_train = target_train
    else:
        model_target_train = y_scaler.transform(target_train)

    incremental_model.partial_fit(norm_data_train, model_target_train)
    return incremental_model


def predict_svr(model, x_scaler, y_scaler, data_test):
    norm_data_test = x_scaler.transform(data_test)
    pred_var = model.predict(norm_data_test)
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
    return mae, ncls, eval_var


def is_valid_test_record(record):
    return len(record["data"]) >= TEST_HOURS * 2


def init_mode_result():
    return {
        experiment["name"]: {
            "mae": [],
            "ncls": [],
            "epsilon": None,
            "files": [],
        }
        for experiment in EXPERIMENTS
    }


def run_task_add(task_name, task_config, add_drifter):
    pre_drifters = [drifter for drifter in DRIFTERS if drifter != add_drifter]
    previous_records = load_records(pre_drifters, task_config)
    additional_records = load_records([add_drifter], task_config)
    test_records = [record for record in additional_records if is_valid_test_record(record)]

    if not test_records:
        return {
            "task": task_name,
            "add": add_drifter,
            "pre": pre_drifters,
            "skipped": True,
            "reason": f"No {add_drifter} files have at least {TEST_END} expanded rows",
        }

    data_previous, target_previous = concatenate(previous_records)
    data_incremental, target_incremental = concatenate(
        test_records,
        slice(0, INCREMENTAL_END),
    )

    result = {
        "task": task_name,
        "add": add_drifter,
        "pre": pre_drifters,
        "pre_dirs": [str(expansion_dir(drifter)) for drifter in pre_drifters],
        "add_dir": str(expansion_dir(add_drifter)),
        "add_original_dir": str(original_dir(add_drifter)),
        "dir_name": str(prediction_dir(add_drifter, task_config)),
        "selected_features": task_config["selected_features"],
        "input_features": task_config["selected_features"][:N_FEATURES],
        "target_features": task_config["selected_features"][N_FEATURES:N_FEATURES + 2],
        "skipped": False,
        "previous_file_count": len(previous_records),
        "additional_file_count": len(additional_records),
        "test_file_count": len(test_records),
        "baseline_mae": [],
        "baseline_ncls": [],
        "modes": {mode: init_mode_result() for mode in MODES},
    }

    for record in test_records:
        data_test = record["data"][TEST_START:TEST_END, :]
        start_point, model_var, obs_var = load_observation(record["original_file"])
        if len(data_test) != len(obs_var):
            raise ValueError(
                f"Length mismatch in test file {record['file']}: "
                f"data={len(data_test)}, obs={len(obs_var)}"
            )

        baseline_mae, baseline_ncls, _ = evaluate(
            start_point,
            model_var,
            obs_var,
            model_var,
            add_model_var=False,
        )
        result["baseline_mae"].append(baseline_mae)
        result["baseline_ncls"].append(baseline_ncls)

    for mode in MODES:
        incremental = mode == "incremental learning"
        for experiment in EXPERIMENTS:
            epsilon = experiment["epsilon"]

            model, x_scaler, y_scaler = fit_svr(
                data_previous,
                target_previous,
                epsilon,
                experiment["scale_target"],
            )
            if incremental:
                model = incremental_fit(
                    model,
                    x_scaler,
                    y_scaler,
                    data_incremental,
                    target_incremental,
                )

            exp_result = result["modes"][mode][experiment["name"]]
            exp_result["epsilon"] = epsilon

            if incremental and SAVE_INCREMENTAL_PREDICTIONS:
                prediction_dir(add_drifter, task_config).mkdir(parents=True, exist_ok=True)
                save_prediction_index(prediction_dir(add_drifter, task_config),
                                      [(i, r["file"].name) for i, r in enumerate(test_records)],
                                      TEST_HOURS, task_name, task_config["selected_features"][:N_FEATURES])

            for file_index, record in enumerate(test_records):
                data_test = record["data"][TEST_START:TEST_END, :]
                start_point, model_var, obs_var = load_observation(
                    record["original_file"]
                )
                pred_var = predict_svr(model, x_scaler, y_scaler, data_test)
                mae, ncls, eval_var = evaluate(
                    start_point,
                    model_var,
                    obs_var,
                    pred_var,
                    task_config["add_model_var"],
                )

                exp_result["mae"].append(mae)
                exp_result["ncls"].append(ncls)
                exp_result["files"].append(
                    {
                        "file": record["file"].name,
                        "epsilon": epsilon,
                        "mae": mae,
                        "ncls": ncls,
                    }
                )

                if incremental and SAVE_INCREMENTAL_PREDICTIONS:
                    output_file = (
                        prediction_dir(add_drifter, task_config)
                        / f"svr{file_index}.csv"
                    )
                    np.savetxt(output_file, eval_var, delimiter=",")

    return result


def mean_std(values):
    values = np.array(values, dtype=float)
    return float(np.mean(values)), float(np.std(values))


def format_mean_std(label, mae_values, ncls_values):
    mae_mean, mae_std = mean_std(mae_values)
    ncls_mean, ncls_std = mean_std(ncls_values)
    return (
        f"{label}: MAE={mae_mean:.10f} +/- {mae_std:.10f}, "
        f"NCLS={ncls_mean:.10f} +/- {ncls_std:.10f}"
    )


def write_results(results):
    lines = [
        "Expanded incremental SVR fixed standardized target epsilon 0.1",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Script: {Path(__file__).name}",
        f"Expanded data dirs: {DATA_PATHS}",
        f"Target original data dir: {args.original_data_path}",
        f"Input features: first {N_FEATURES} columns from each task's selected_features",
        "Targets: last 2 columns from each task's selected_features",
        f"Test window: rows {TEST_START}:{TEST_END}",
        f"Incremental update window: rows 0:{INCREMENTAL_END}",
        "SVR config: standardized target, epsilon=0.1",
        "",
    ]

    for result in results:
        lines.append(f"Task: {result['task']}")
        lines.append(f"Add: {result['add']}")
        lines.append(f"Pre: {', '.join(result['pre'])}")

        if result["skipped"]:
            lines.append(f"SKIP: {result['reason']}")
            lines.append("")
            continue

        lines.append(f"Pre dirs: {', '.join(result['pre_dirs'])}")
        lines.append(f"Add dir: {result['add_dir']}")
        lines.append(f"Add original dir: {result['add_original_dir']}")
        lines.append(f"Dir name: {result['dir_name']}")
        lines.append(f"Selected features: {result['selected_features']}")
        lines.append(f"Input features: {result['input_features']}")
        lines.append(f"Target features: {result['target_features']}")
        lines.append(f"Previous files: {result['previous_file_count']}")
        lines.append(f"Additional files: {result['additional_file_count']}")
        lines.append(f"Test files: {result['test_file_count']}")

        for mode in MODES:
            lines.append(mode)
            lines.append(
                format_mean_std(
                    "Baseline",
                    result["baseline_mae"],
                    result["baseline_ncls"],
                )
            )

            for experiment in EXPERIMENTS:
                exp_result = result["modes"][mode][experiment["name"]]
                lines.append(
                    format_mean_std(
                        experiment["name"],
                        exp_result["mae"],
                        exp_result["ncls"],
                    )
                )
            lines.append("")

        lines.append("Per-file results")
        lines.append("mode,file,experiment,epsilon,mae,ncls")
        for mode in MODES:
            for experiment in EXPERIMENTS:
                exp_result = result["modes"][mode][experiment["name"]]
                for file_result in exp_result["files"]:
                    lines.append(
                        f"{mode},{file_result['file']},{experiment['name']},"
                        f"{file_result['epsilon']},{file_result['mae']:.10f},"
                        f"{file_result['ncls']:.10f}"
                    )
        lines.append("")

    RESULT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    results = []
    for task_name, task_config in TASKS.items():
        for add_drifter in ["target"]:
            print(f"Running {task_name} / add={add_drifter}")
            results.append(run_task_add(task_name, task_config, add_drifter))

    write_results(results)
    print(f"Saved results to {RESULT_FILE}")


if __name__ == "__main__":
    main()
