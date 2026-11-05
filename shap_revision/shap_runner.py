"""Shared body of feature_expansion/shap_prediction.py and shap_correction.py.
The original 30-feature training and SHAP calculations are preserved.
"""
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import SGDRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import pickle
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from geopy.distance import geodesic
import os
import sys
import random
import shap
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

import csv
from publication_utils import arguments, output_directory, csv_files, save_file_order


def save_csv_ranking(path, dataset_label, task, model, shap_idx, shap_mean):
    """Store the original full ranking; rank <= 10 is selected for counting."""
    with open(path, 'a', newline='') as stream:
        writer = csv.writer(stream)
        for rank, feature in enumerate(shap_idx, 1):
            writer.writerow((dataset_label, task, 30, model, rank,
                             int(feature), float(shap_mean[feature])))


def run_shap_experiments(analysis_name, target_columns):
    """Original 30-feature SHAP experiment for one dataset and one task."""
    args = arguments('shap')
    if not args.dataset_label or '/' in args.dataset_label or '\\' in args.dataset_label:
        raise ValueError('dataset-label must be a nonempty filename component')
    output_directory(args)
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    original = args.dataset_label
    pre = args.data_path

    #load csv files
    previous = csv_files(pre, args.sort_files, args.file_order)
    if not previous:
        raise ValueError('No trajectory CSV files found in data_path')
    save_file_order('file_order.csv', previous)
    result_csv = f'shap_{analysis_name}_{original}.csv'
    with open(result_csv, 'w', newline='') as stream:
        csv.writer(stream).writerow(('dataset_label', 'task', 'n_features', 'model',
                                     'rank', 'feature_index', 'shap_mean'))

    test_hours = 48
    timestep = 6


    #make dataset
    for i in range(len(previous)):
        filename = pre + '/' + previous[i]
        file = np.loadtxt(filename, skiprows=1, delimiter=',', ndmin=2)
        if file.shape[0] < timestep or file.shape[1] < 34:
            raise ValueError(f'{filename}: expected at least 6 rows and 34 columns')

        data=file[:, :30]
        target=file[:, target_columns]
        if not np.isfinite(data).all() or not np.isfinite(target).all():
            raise ValueError(f'{filename}: non-finite feature or target values')

        globals()[f'previous_data_variable{i}']=data
        globals()[f'previous_target_variable{i}']=target


    # pretrained model
    # make dataset
    data_train = np.zeros((1, 30))
    target_train = np.zeros((1, 2))
    for i in range(0,len(previous)):
        data_train = np.append(data_train, globals()[f'previous_data_variable{i}'], axis=0)
        target_train = np.append(target_train, globals()[f'previous_target_variable{i}'], axis=0)

    data_train = np.delete(data_train, 0, axis=0)
    target_train = np.delete(target_train, 0, axis=0)
    target_train_x=np.reshape(target_train[:,0], (-1,))
    target_train_y=np.reshape(target_train[:,1], (-1,))

    scaler = StandardScaler()
    scaler.fit(data_train)
    joblib.dump(scaler, f'scaler_{analysis_name}_{original}.pkl')
    norm_data_train = scaler.transform(data_train)

    # make dataset for lstm
    lstm_data_train = np.zeros((1, timestep, 30))
    lstm_target_train = np.zeros((1,2))
    for i in range(0, len(previous)):
        data = globals()[f'previous_data_variable{i}']
        data = scaler.transform(data)
        target = globals()[f'previous_target_variable{i}']

        for j in range(len(data) - timestep + 1):
            lstm_data_train = np.append(lstm_data_train, np.reshape(data[j:j + timestep, :], (1, timestep, 30)), axis=0)

        lstm_target_train = np.append(lstm_target_train, target[timestep - 1:, :], axis=0)
    lstm_data_train=np.delete(lstm_data_train, 0, axis=0)
    lstm_target_train=np.delete(lstm_target_train, 0, axis=0)
    lstm_target_train_x=np.reshape(lstm_target_train[:,0], (-1,1))
    lstm_target_train_y=np.reshape(lstm_target_train[:,1], (-1,1))


    # LightGBM
    lgbm_x=LGBMRegressor()
    lgbm_y=LGBMRegressor()

    lgbm_x.fit(norm_data_train, target_train_x)
    lgbm_y.fit(norm_data_train, target_train_y)

    # LSTM
    # define model
    lstm_x = tf.keras.Sequential()
    lstm_x.add(layers.LSTM(30))
    lstm_x.add(layers.Dense(2, activation='sigmoid'))
    lstm_x.add(layers.Dense(1))

    lstm_x.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

    # train model
    lstm_x.fit(lstm_data_train, lstm_target_train_x, batch_size=32, epochs=args.epochs, callbacks=[es])


    lstm_y = tf.keras.Sequential()
    lstm_y.add(layers.LSTM(30))
    lstm_y.add(layers.Dense(2, activation='sigmoid'))
    lstm_y.add(layers.Dense(1))

    lstm_y.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

    # train model
    lstm_y.fit(lstm_data_train, lstm_target_train_y, batch_size=32, epochs=args.epochs, callbacks=[es])


    # calculate lgbm shap value
    shap_mean=np.zeros((1,30))
    explainer_x=shap.TreeExplainer(lgbm_x)
    explainer_y=shap.TreeExplainer(lgbm_y)

    shap_values_x=explainer_x.shap_values(norm_data_train)
    shap_values_y=explainer_y.shap_values(norm_data_train)

    shap_x_norm=np.abs(shap_values_x)
    shap_y_norm=np.abs(shap_values_y)

    # SHAP summary
    shap.summary_plot(shap_x_norm, show=False)
    # shap.summary_plot(shap_values_x)
    plt.savefig(f"{analysis_name}_shap_lgbm_x_{original}.png")
    plt.close()

    shap.summary_plot(shap_y_norm, show=False)
    # shap.summary_plot(shap_values_y)
    plt.savefig(f"{analysis_name}_shap_lgbm_y_{original}.png")
    plt.close()

    # best features
    # shap_x_norm=np.abs(shap_values_x)
    # shap_y_norm=np.abs(shap_values_y)

    shap_x_norm=shap_x_norm/np.max(shap_x_norm) if np.max(shap_x_norm) > 0 else np.zeros_like(shap_x_norm)
    shap_y_norm=shap_y_norm/np.max(shap_y_norm) if np.max(shap_y_norm) > 0 else np.zeros_like(shap_y_norm)

    shap_norm=(shap_x_norm+shap_y_norm)/2
    shap_mean=np.mean(shap_norm, axis=0)
    temp=sys.stdout
    sys.stdout = open(f'shap_{analysis_name}_{original}.txt', 'a')
    shap_idx=np.argsort(-shap_mean)
    print('lgbm')
    print(shap_idx)
    print(shap_mean[shap_idx])
    print(shap_idx[:10])
    print(shap_mean[shap_idx[:10]])
    sys.stdout.close()
    sys.stdout=temp
    save_csv_ranking(result_csv, original, analysis_name, 'lgbm', shap_idx, shap_mean)


    # calculate lstm shap value
    n_samples = lstm_data_train.shape[0]

    # 행 인덱스 10% 무작위 추출(중복 없이)
    idx = np.random.choice(n_samples, max(1, int(n_samples*0.1)), replace=False)

    # 해당 행만 슬라이싱 → background 데이터
    background = lstm_data_train[idx]

    explainer_x=shap.GradientExplainer(lstm_x, background)
    explainer_y=shap.GradientExplainer(lstm_y, background)

    shap_values_x=explainer_x.shap_values(lstm_data_train)
    shap_values_y=explainer_y.shap_values(lstm_data_train)

    shap_values_x=np.reshape(shap_values_x, (-1, 6, 30))
    shap_values_y=np.reshape(shap_values_y, (-1, 6, 30))

    shap_x_norm=np.abs(shap_values_x)
    shap_y_norm=np.abs(shap_values_y)

    shap_x_norm=np.sum(shap_x_norm, axis=1)
    shap_y_norm=np.sum(shap_y_norm, axis=1)
    # SHAP summary
    # shap.summary_plot(shap_values_x, lstm_data_train)
    shap.summary_plot(shap_x_norm, show=False)
    # shap.summary_plot(shap_x_norm, norm_data_train)
    plt.savefig(f"{analysis_name}_shap_lstm_x_{original}.png")
    plt.close()

    # shap.summary_plot(shap_values_y, lstm_data_train)
    shap.summary_plot(shap_y_norm, show=False)
    # shap.summary_plot(shap_y_norm, norm_data_train)
    plt.savefig(f"{analysis_name}_shap_lstm_y_{original}.png")
    plt.close()

    # best features
    # shap_values_x=np.reshape(shap_values_x, (-1, 6, 30))
    # shap_values_y=np.reshape(shap_values_y, (-1, 6, 30))

    # shap_x_norm=np.abs(shap_values_x)
    # shap_y_norm=np.abs(shap_values_y)

    # shap_x_norm=np.sum(shap_x_norm, axis=1)
    # shap_y_norm=np.sum(shap_y_norm, axis=1)

    shap_x_norm=shap_x_norm/np.max(shap_x_norm) if np.max(shap_x_norm) > 0 else np.zeros_like(shap_x_norm)
    shap_y_norm=shap_y_norm/np.max(shap_y_norm) if np.max(shap_y_norm) > 0 else np.zeros_like(shap_y_norm)

    shap_norm=(shap_x_norm+shap_y_norm)/2
    shap_mean=np.mean(shap_norm, axis=0)
    temp=sys.stdout
    sys.stdout = open(f'shap_{analysis_name}_{original}.txt', 'a')
    shap_idx=np.argsort(-shap_mean)
    print('lstm')
    print(shap_idx)
    print(shap_mean[shap_idx])
    print(shap_idx[:10])
    print(shap_mean[shap_idx[:10]])
    sys.stdout.close()
    sys.stdout=temp
    save_csv_ranking(result_csv, original, analysis_name, 'lstm', shap_idx, shap_mean)
