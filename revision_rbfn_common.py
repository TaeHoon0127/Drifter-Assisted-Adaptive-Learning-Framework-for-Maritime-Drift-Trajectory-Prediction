import csv
import glob
import io
import os
import random
import shutil
import subprocess
import tempfile

import numpy as np


FEATURE_COLS = (4, 5, 6, 7, 8, 9)
OBS_COLS = (8, 9, 10, 11)
N_FEATURES = 6
N_OUTPUTS = 2

TIMESTEP = 6
ALIGNMENT_OFFSET = 3
TEST_START_INDEX = TIMESTEP - 1 + ALIGNMENT_OFFSET
EARTH_RADIUS_KM = 6371.0088
NCLS_DISTANCE = "haversine"

# Published RBFN parameters from the paper. Remaining WEKA settings use defaults.
WEKA_RBFN_CLASSIFIER = 'weka.classifiers.functions.RBFRegressor'
WEKA_DECIMAL_PLACES = 2
WEKA_RBFN_OPTIONS = (
    '-N', '2',
    '-R', '2',
    '-C', '2',
    '-L', '1.0E-6',
    '-num-decimal-places', str(WEKA_DECIMAL_PLACES),
)


def path_from_script(script_dir, path):
    return os.path.normpath(os.path.join(script_dir, path))


def set_random_seed():
    random.seed(42)
    np.random.seed(42)


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def haversine_km(point_a, point_b):
    lat1, lon1 = np.radians(point_a)
    lat2, lon2 = np.radians(point_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(hav))


def distance_km(a, b):
    if NCLS_DISTANCE == "geodesic":
        from geopy.distance import geodesic
        return geodesic(a, b).km
    return haversine_km(a, b)


def calculate_ncls(obs, pred):
    obs_loc = obs.copy()
    pred_loc = pred.copy()
    obs_loc[:, [0, 1]] = obs_loc[:, [1, 0]]
    pred_loc[:, [0, 1]] = pred_loc[:, [1, 0]]

    distances = []
    for i in range(1, len(obs_loc)):
        distances.append(distance_km(obs_loc[i], pred_loc[i]))
    distances = np.array(distances)

    segment_lengths = []
    for i in range(len(obs_loc) - 1):
        segment_lengths.append(distance_km(obs_loc[i], obs_loc[i + 1]))
    segment_lengths = np.array(segment_lengths)

    cumulative_lengths = [0]
    for length in segment_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + length)
    cumulative_lengths = np.array(cumulative_lengths)
    cumulative_lengths = np.delete(cumulative_lengths, 0, axis=0)

    total_length = np.sum(cumulative_lengths)
    if total_length == 0:
        return np.nan

    score = np.sum(distances) / total_length
    if score > 1:
        return 0

    return 1 - score


def load_previous_data(dataset_path, target_cols):
    filenames = sorted(
        name for name in os.listdir(dataset_path) if name.lower().endswith('.csv')
    )
    previous_data = []
    previous_target = []

    for filename in filenames:
        file_path = os.path.join(dataset_path, filename)
        file = np.loadtxt(
            file_path,
            skiprows=1,
            delimiter=',',
            usecols=FEATURE_COLS + target_cols,
        )
        file = np.atleast_2d(file)
        previous_data.append(file[:, :N_FEATURES])
        previous_target.append(file[:, N_FEATURES:N_FEATURES + N_OUTPUTS])

    return filenames, previous_data, previous_target


def make_locations(start_point, variables):
    start_point = np.reshape(start_point, (1, 2))
    return np.vstack((start_point, start_point + np.cumsum(variables, axis=0)))


def load_observation(dataset_path, filename):
    file_path = os.path.join(dataset_path, filename)
    file = np.loadtxt(
        file_path,
        skiprows=TIMESTEP + ALIGNMENT_OFFSET,
        delimiter=',',
        usecols=OBS_COLS,
    )
    file = np.atleast_2d(file)
    start_point = np.loadtxt(
        file_path,
        skiprows=TEST_START_INDEX,
        max_rows=1,
        delimiter=',',
        usecols=(0, 1),
    )
    model_var = file[:, :2]
    obs_var = file[:, 2:]
    return start_point, model_var, obs_var


def find_java():
    java_bin = os.environ.get('JAVA_BIN') or shutil.which('java')
    if java_bin:
        return java_bin

    raise RuntimeError(
        'Java runtime was not found. Install Java and make `java` available on '
        'PATH, or set JAVA_BIN to the Java executable.'
    )


def find_weka_classpath(script_dir):
    classpath = os.environ.get('WEKA_CLASSPATH')
    if classpath:
        return classpath

    weka_jar = os.environ.get('WEKA_JAR')
    if weka_jar:
        if os.path.exists(weka_jar):
            return weka_jar
        raise RuntimeError(f'WEKA_JAR does not exist: {weka_jar}')

    search_dirs = [
        script_dir,
        os.path.dirname(script_dir),
        os.getcwd(),
        os.path.expanduser('~'),
        '/usr/share/java',
        '/usr/local/share/java',
    ]
    patterns = (
        'weka.jar',
        'weka-*.jar',
        'weka-stable*.jar',
        'weka-dev*.jar',
        'RBFNetwork*.jar',
    )
    candidates = []
    for search_dir in search_dirs:
        for pattern in patterns:
            candidates.extend(glob.glob(os.path.join(search_dir, pattern)))

    candidates = sorted(set(candidates))
    if candidates:
        return os.pathsep.join(candidates)

    raise RuntimeError(
        'WEKA jar was not found. Set WEKA_JAR=/path/to/weka.jar, or set '
        'WEKA_CLASSPATH to include weka.jar and, if needed, RBFNetwork.jar '
        'for weka.classifiers.functions.RBFRegressor.'
    )


def format_arff_value(value):
    if np.isfinite(value):
        return f'{float(value):.17g}'
    return '?'


def write_arff(path, relation, x, y):
    with open(path, 'w') as output:
        print(f'@relation {relation}', file=output)
        for idx in range(N_FEATURES):
            print(f'@attribute f{idx} numeric', file=output)
        print('@attribute target numeric', file=output)
        print('@data', file=output)
        for features, target in zip(x, y):
            values = [format_arff_value(value) for value in features]
            values.append(format_arff_value(target))
            print(','.join(values), file=output)


def parse_weka_value(value):
    value = value.strip()
    if ':' in value:
        value = value.split(':', 1)[1]
    return float(value)


def parse_csv_predictions(stdout):
    predicted_idx = None
    predictions = []
    reader = csv.reader(io.StringIO(stdout))

    for row in reader:
        if not row:
            continue

        row = [field.strip() for field in row]
        lower_row = [field.lower() for field in row]
        if 'predicted' in lower_row:
            predicted_idx = lower_row.index('predicted')
            continue

        if predicted_idx is None:
            continue
        if len(row) <= predicted_idx or not row[0].isdigit():
            continue

        predictions.append(parse_weka_value(row[predicted_idx]))

    return predictions


def parse_plain_text_predictions(stdout):
    predictions = []
    for line in stdout.splitlines():
        tokens = line.split()
        if len(tokens) < 3 or not tokens[0].isdigit():
            continue
        predictions.append(parse_weka_value(tokens[2]))

    return predictions


def run_weka_command(java_bin, classpath, train_path, test_path, use_csv=True):
    cmd = [
        java_bin,
        '-cp',
        classpath,
        WEKA_RBFN_CLASSIFIER,
        '-t',
        train_path,
        '-T',
        test_path,
        '-c',
        'last',
    ]

    if use_csv:
        cmd.extend([
            '-classifications',
            (
                'weka.classifiers.evaluation.output.prediction.CSV '
                f'-decimals {WEKA_DECIMAL_PLACES}'
            ),
        ])
    else:
        cmd.extend(['-p', '0'])

    cmd.extend(WEKA_RBFN_OPTIONS)
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    return result


def summarize_weka_output(result):
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    stdout_sample = stdout[:2000] if stdout else '(empty)'
    stderr_sample = stderr[:2000] if stderr else '(empty)'
    return (
        f'return code: {result.returncode}\n'
        f'stdout sample:\n{stdout_sample}\n'
        f'stderr sample:\n{stderr_sample}'
    )


def predict_one_target(java_bin, classpath, train_x, train_y, test_x, test_y, tmp_dir, target_idx):
    predictions = []
    train_path = os.path.join(tmp_dir, f'train_target_{target_idx}.arff')
    test_path = os.path.join(tmp_dir, f'test_target_{target_idx}.arff')

    write_arff(train_path, f'rbfn_train_target_{target_idx}', train_x, train_y)
    write_arff(test_path, f'rbfn_test_target_{target_idx}', test_x, test_y)

    result = run_weka_command(java_bin, classpath, train_path, test_path, use_csv=True)
    if result.returncode == 0:
        predictions = parse_csv_predictions(result.stdout)
        if len(predictions) == len(test_x):
            return np.array(predictions)

    fallback = run_weka_command(java_bin, classpath, train_path, test_path, use_csv=False)
    if fallback.returncode == 0:
        predictions = parse_plain_text_predictions(fallback.stdout)
        if len(predictions) == len(test_x):
            return np.array(predictions)

    raise RuntimeError(
        f'Expected {len(test_x)} WEKA predictions for target {target_idx}, '
        f'got {len(predictions)}.\n'
        'CSV WEKA output:\n'
        f'{summarize_weka_output(result)}\n'
        'Plain-text fallback WEKA output:\n'
        f'{summarize_weka_output(fallback)}'
    )


def predict_with_weka_rbfn(java_bin, classpath, train_x, train_y, test_x, test_y):
    with tempfile.TemporaryDirectory() as tmp_dir:
        pred_x = predict_one_target(
            java_bin,
            classpath,
            train_x,
            train_y[:, 0],
            test_x,
            test_y[:, 0],
            tmp_dir,
            0,
        )
        pred_y = predict_one_target(
            java_bin,
            classpath,
            train_x,
            train_y[:, 1],
            test_x,
            test_y[:, 1],
            tmp_dir,
            1,
        )

    return np.column_stack((pred_x, pred_y))


def append_results(output_path, dataset, metrics):
    with open(output_path, 'a') as output:
        print('Test:', dataset, file=output)
        print(
            'Baseline:',
            np.mean(metrics['base_mae']),
            np.mean(metrics['base_ncls']),
            file=output,
        )
        print(
            'RBFN:',
            np.mean(metrics['rbfn_mae']),
            np.mean(metrics['rbfn_ncls']),
            file=output,
        )


def run_single_dataset(
    script_dir,
    dataset,
    target_cols,
    output_path,
    apply_correction,
    java_bin,
    classpath,
):
    dataset_path = path_from_script(script_dir, dataset)
    filenames, previous_data, previous_target = load_previous_data(
        dataset_path,
        target_cols,
    )
    if len(filenames) < 2:
        raise ValueError(f'{dataset} requires at least two CSV files.')

    metrics = {
        'base_mae': [],
        'rbfn_mae': [],
        'base_ncls': [],
        'rbfn_ncls': [],
    }

    for k, filename in enumerate(filenames):
        train_indices = [i for i in range(len(filenames)) if i != k]
        data_train = np.concatenate([previous_data[i] for i in train_indices], axis=0)
        target_train = np.concatenate(
            [previous_target[i] for i in train_indices],
            axis=0,
        )

        data_test = previous_data[k][TEST_START_INDEX:, :]
        target_test = previous_target[k][TEST_START_INDEX:, :]
        pred_var = predict_with_weka_rbfn(
            java_bin,
            classpath,
            data_train,
            target_train,
            data_test,
            target_test,
        )

        start_point, model_var, obs_var = load_observation(dataset_path, filename)
        model_loc = make_locations(start_point, model_var)
        obs_loc = make_locations(start_point, obs_var)

        metrics['base_mae'].append(mean_absolute_error(obs_var, model_var))
        metrics['base_ncls'].append(calculate_ncls(obs_loc, model_loc))

        if apply_correction:
            rbfn_var = model_var + pred_var
        else:
            rbfn_var = pred_var

        rbfn_loc = make_locations(start_point, rbfn_var)
        metrics['rbfn_mae'].append(mean_absolute_error(obs_var, rbfn_var))
        metrics['rbfn_ncls'].append(calculate_ncls(obs_loc, rbfn_loc))

    append_results(output_path, dataset, metrics)


def run_experiment(script_dir, datasets, target_cols, output_filename, apply_correction,
                   decimal_places=2, distance="haversine"):
    global WEKA_DECIMAL_PLACES, WEKA_RBFN_OPTIONS, NCLS_DISTANCE
    WEKA_DECIMAL_PLACES = decimal_places
    WEKA_RBFN_OPTIONS = WEKA_RBFN_OPTIONS[:-1] + (str(decimal_places),)
    NCLS_DISTANCE = distance
    set_random_seed()
    java_bin = find_java()
    classpath = find_weka_classpath(script_dir)
    output_path = path_from_script(script_dir, output_filename)
    for dataset in datasets:
        run_single_dataset(
            script_dir,
            dataset,
            target_cols,
            output_path,
            apply_correction,
            java_bin,
            classpath,
        )
