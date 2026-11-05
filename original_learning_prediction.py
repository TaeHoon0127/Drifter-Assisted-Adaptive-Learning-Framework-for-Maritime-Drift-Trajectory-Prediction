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

from publication_utils import arguments, output_directory, csv_files, save_file_order, StandardizedTargetSVR

args = arguments()
output_directory(args)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

pre = args.data_path

#load csv files
previous = csv_files(pre, args.sort_files, args.file_order)
save_file_order('file_order.csv', previous)
if len(previous) < 2:
    raise ValueError('LOO requires at least two trajectory CSV files')

test_hours = 48
timestep = 6


def calculate_ncls(obs, pred):
    obs_loc=obs.copy()
    pred_loc=pred.copy()
    obs_loc[:, [0,1]]=obs_loc[:, [1,0]]
    pred_loc[:, [0,1]]=pred_loc[:, [1,0]]
    d=[]
    for i in range(1, len(obs_loc)):
        d.append(geodesic(obs_loc[i], pred_loc[i]).km)
    d=np.array(d)
    
    dl=[]
    for i in range(len(obs_loc)-1):
        dl.append(geodesic(obs_loc[i], obs_loc[i+1]).km)
    dl=np.array(dl)
    
    l=[0]
    for i in range(len(dl)):
        l.append(l[i]+dl[i])
    l=np.array(l)
    l=np.delete(l,0,axis=0)
    
    denominator = np.sum(l)
    if denominator == 0:
        return np.nan
    s=np.sum(d)/denominator
    
    ss=0
    if s>1:
        ss=0
    else:
        ss=1-s
    
    return ss
    
#make dataset
for i in range(len(previous)):
    filename = pre + '/' + previous[i]
    file = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=(4,5,6,7,8,9,10,11))

    data=file[:, :6]
    target=file[:, 6:]

    globals()[f'previous_data_variable{i}']=data
    globals()[f'previous_target_variable{i}']=target


base_mae=[]
linear_mae=[]
svr_mae=[]
mlp_mae=[]
lgbm_mae=[]
lstm_mae=[]
base_ncls=[]
linear_ncls=[]
svr_ncls=[]
mlp_ncls=[]
lgbm_ncls=[]
lstm_ncls=[]
for k in range(len(previous)):
    # pretrained model
    # make dataset
    data_train = np.zeros((1, 6))
    target_train = np.zeros((1, 2))
    for i in range(0,k):
        data_train = np.append(data_train, globals()[f'previous_data_variable{i}'], axis=0)
        target_train = np.append(target_train, globals()[f'previous_target_variable{i}'], axis=0)
    for i in range(k+1,len(previous)):
        data_train = np.append(data_train, globals()[f'previous_data_variable{i}'], axis=0)
        target_train = np.append(target_train, globals()[f'previous_target_variable{i}'], axis=0)

    data_train = np.delete(data_train, 0, axis=0)
    target_train = np.delete(target_train, 0, axis=0)
    target_train_x=np.reshape(target_train[:,0], (-1,))
    target_train_y=np.reshape(target_train[:,1], (-1,))

    scaler = StandardScaler()
    scaler.fit(data_train)
    joblib.dump(scaler, 'scaler.pkl')
    norm_data_train = scaler.transform(data_train)

    # make dataset for lstm
    lstm_data_train = np.zeros((1, timestep, 6))
    lstm_target_train = np.zeros((1,2))
    for i in range(0, k):
        data=globals()[f'previous_data_variable{i}']
        data=scaler.transform(data)
        target=globals()[f'previous_target_variable{i}']

        for j in range(len(data)-timestep+1):
            lstm_data_train=np.append(lstm_data_train, np.reshape(data[j:j+timestep, :], (1, timestep, 6)), axis=0)

        lstm_target_train=np.append(lstm_target_train, target[timestep-1:, :], axis=0)
    for i in range(k+1, len(previous)):
        data = globals()[f'previous_data_variable{i}']
        data = scaler.transform(data)
        target = globals()[f'previous_target_variable{i}']

        for j in range(len(data) - timestep + 1):
            lstm_data_train = np.append(lstm_data_train, np.reshape(data[j:j + timestep, :], (1, timestep, 6)), axis=0)

        lstm_target_train = np.append(lstm_target_train, target[timestep - 1:, :], axis=0)
    lstm_data_train=np.delete(lstm_data_train, 0, axis=0)
    lstm_target_train=np.delete(lstm_target_train, 0, axis=0)


    # Linear
    linear = models.Sequential()
    linear.add(layers.Input(shape=(6, )))
    linear.add(layers.Dense(2))
    linear.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    linear.fit(norm_data_train, target_train, batch_size=32, epochs=args.epochs, callbacks=[es])

    # SVR
    svr = StandardizedTargetSVR()
    svr.partial_fit(norm_data_train, target_train)

    # MLP
    mlp = models.Sequential()
    mlp.add(layers.Input(shape=(6, )))
    mlp.add(layers.Dense(2, activation='sigmoid'))
    mlp.add(layers.Dense(2))
    mlp.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

    # train model
    mlp.fit(norm_data_train, target_train, batch_size=32, epochs=args.epochs, callbacks=[es])

    # LightGBM
    lgbm_x=LGBMRegressor()
    lgbm_y=LGBMRegressor()

    lgbm_x.fit(norm_data_train, target_train_x)
    lgbm_y.fit(norm_data_train, target_train_y)

    # LSTM
    # define model
    lstm = tf.keras.Sequential()
    lstm.add(layers.LSTM(6))
    lstm.add(layers.Dense(2, activation='sigmoid'))
    lstm.add(layers.Dense(2))

    lstm.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

    # train model
    lstm.fit(lstm_data_train, lstm_target_train, batch_size=32, epochs=args.epochs, callbacks=[es])


    # test models performance
    # make dataset
    data_test = globals()[f'previous_data_variable{k}']
    data_test = data_test[timestep-1+3:, :] # 특징확장 데이터와 테스트 개수 맞춤
    norm_data_test = scaler.transform(data_test)

    # make dataset for lstm
    lstm_data_test = np.zeros((1, timestep, 6))

    data = globals()[f'previous_data_variable{k}']
    data = scaler.transform(data)
    data = data[3:, :] # 특징확장 데이터와 테스트 개수 맞춤

    for j in range(len(data) - timestep + 1):
        lstm_data_test = np.append(lstm_data_test, np.reshape(data[j:j + timestep, :], (1, timestep, 6)), axis=0)

    lstm_data_test = np.delete(lstm_data_test, 0, axis=0)

    # observation
    filename = pre + '/' + previous[k] # 바꿀 부분
    file = np.loadtxt(filename, skiprows=timestep+3, delimiter=',', usecols=(8,9,10,11)) # 특징확장 데이터와 테스트 개수 맞춤
    start_point = np.loadtxt(filename, skiprows=timestep+2, max_rows=1, delimiter=',', usecols=(0, 1))
    start_point = np.reshape(start_point, (1,2))
    model_var=file[:, :2]
    obs_var = file[:, 2:]

    # calculate location
    model_loc=start_point
    obs_loc=start_point
    for j in range(len(obs_var)):
        model_loc=np.append(model_loc, np.reshape(model_loc[j]+model_var[j], (1,2)), axis=0)
        obs_loc=np.append(obs_loc, np.reshape(obs_loc[j]+obs_var[j], (1,2)), axis=0)

    # Baseline
    base_mae.append(mean_absolute_error(obs_var, model_var))
    base_ncls.append(calculate_ncls(obs_loc, model_loc))

    # Linear
    pred_var=linear.predict(norm_data_test)
    # np.savetxt(f'4pred_kaist_svp/linear.csv', pred_var, delimiter=',')
    linear_mae.append(mean_absolute_error(obs_var, pred_var))
    
    linear_loc=start_point
    for j in range(len(pred_var)):
        linear_loc=np.append(linear_loc, np.reshape(linear_loc[j]+pred_var[j], (1,2)), axis=0)
        
    linear_ncls.append(calculate_ncls(obs_loc, linear_loc))

    # SVR
    pred_var=svr.predict(norm_data_test)
    # np.savetxt(f'4pred_kaist_svp/svr.csv', pred_var, delimiter=',')
    svr_mae.append(mean_absolute_error(obs_var, pred_var))
    
    svr_loc=start_point
    for j in range(len(pred_var)):
        svr_loc=np.append(svr_loc, np.reshape(svr_loc[j]+pred_var[j], (1,2)), axis=0)
        
    svr_ncls.append(calculate_ncls(obs_loc, svr_loc))

    # MLP
    pred_var=mlp.predict(norm_data_test)
    # np.savetxt(f'4pred_kaist_svp/mlp.csv', pred_var, delimiter=',')
    mlp_mae.append(mean_absolute_error(obs_var, pred_var))
    
    mlp_loc=start_point
    for j in range(len(pred_var)):
        mlp_loc=np.append(mlp_loc, np.reshape(mlp_loc[j]+pred_var[j], (1,2)), axis=0)
        
    mlp_ncls.append(calculate_ncls(obs_loc, mlp_loc))

    # LGBM
    pred_x = np.reshape(lgbm_x.predict(norm_data_test), (-1, 1))
    pred_y = np.reshape(lgbm_y.predict(norm_data_test), (-1,1))
    pred_var=np.append(pred_x, pred_y, axis=1)
    # np.savetxt(f'4pred_kaist_svp/lgbm.csv', pred_var, delimiter=',')
    lgbm_mae.append(mean_absolute_error(obs_var, pred_var))
    
    lgbm_loc=start_point
    for j in range(len(pred_var)):
        lgbm_loc=np.append(lgbm_loc, np.reshape(lgbm_loc[j]+pred_var[j], (1,2)), axis=0)
        
    lgbm_ncls.append(calculate_ncls(obs_loc, lgbm_loc))

    # LSTM
    pred_var=lstm.predict(lstm_data_test)
    # np.savetxt(f'4pred_kaist_svp/lstm.csv', pred_var, delimiter=',')
    lstm_mae.append(mean_absolute_error(obs_var, pred_var))
    
    lstm_loc=start_point
    for j in range(len(pred_var)):
        lstm_loc=np.append(lstm_loc, np.reshape(lstm_loc[j]+pred_var[j], (1,2)), axis=0)
        
    lstm_ncls.append(calculate_ncls(obs_loc, lstm_loc))


temp=sys.stdout
sys.stdout = open('original_learning_prediction.txt', 'a') # 바꿀 부분
print('Test:', pre)
print('Baseline:', np.mean(base_mae), np.mean(base_ncls))
print('Linear:', np.mean(linear_mae), np.mean(linear_ncls))
print('SVR:', np.mean(svr_mae), np.mean(svr_ncls))
print('MLP:', np.mean(mlp_mae), np.mean(mlp_ncls))
print('LGBM:', np.mean(lgbm_mae), np.mean(lgbm_ncls))
print('LSTM:', np.mean(lstm_mae), np.mean(lstm_ncls))
print('Average:', np.mean([np.mean(linear_mae), np.mean(svr_mae), np.mean(mlp_mae), np.mean(lgbm_mae), np.mean(lstm_mae)]), np.mean([np.mean(linear_ncls), np.mean(svr_ncls), np.mean(mlp_ncls), np.mean(lgbm_ncls), np.mean(lstm_ncls)]))
sys.stdout.close()
sys.stdout=temp
