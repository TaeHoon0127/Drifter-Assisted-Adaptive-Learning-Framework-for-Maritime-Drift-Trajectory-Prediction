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

from publication_utils import arguments, output_directory, csv_files, StandardizedTargetSVR, save_prediction_index

args = arguments()
if not args.pretrain_data_path or args.original_data_path is None:
    raise ValueError('--pretrain-data-path SOURCE_1 SOURCE_2 and --original-data-path are required')
if args.test_hours <= 8:
    raise ValueError('test-hours must exceed the sequence/feature history')
output_directory(args)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

pre1, pre2 = args.pretrain_data_path
add = args.data_path
add_original = args.original_data_path
dir_name = 'predictions'
os.makedirs(dir_name, exist_ok=True)

#load csv files
additional = csv_files(add)
previous1 = csv_files(pre1)
previous2 = csv_files(pre2)

additional.sort()
previous1.sort()
previous2.sort()

test_hours = args.test_hours
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


selected_features=[5,8,9,12,13,20,21,26,28,29,30,31]

#make dataset
for i in range(len(additional)):
    filename = add + '/' + additional[i] # 바꿀 부분
    file = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=selected_features)

    data=file[:, :10]
    target=file[:, 10:12]

    globals()[f'additional_data_variable{i}']=data
    globals()[f'additional_target_variable{i}']=target

for i in range(len(previous1)):
    filename = pre1 + '/' + previous1[i]
    file = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=selected_features)

    data=file[:, :10]
    target=file[:, 10:12]

    globals()[f'previous_data_variable{i}']=data
    globals()[f'previous_target_variable{i}']=target

for i in range(len(previous2)):
    filename = pre2 + '/' + previous2[i]
    file = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=selected_features)

    data=file[:, :10]
    target=file[:, 10:12]

    globals()[f'previous_data_variable{len(previous1)+i}']=data
    globals()[f'previous_target_variable{len(previous1)+i}']=target


# Preserve numbered predictions and save an explicit trajectory mapping.
prediction_pairs = [(i, name) for i, name in enumerate(additional)
                    if len(globals()[f'additional_data_variable{i}']) >= test_hours*2]
if not prediction_pairs:
    raise ValueError('No eligible target trajectories')
save_prediction_index(dir_name, prediction_pairs, test_hours, 'prediction', selected_features[:10])

# pretrained model
# make dataset
data_train = np.zeros((1, 10))
target_train = np.zeros((1, 2))
for i in range(len(previous1)+len(previous2)):
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
lstm_data_train = []
lstm_target_train = np.zeros((1,2))
for i in range(len(previous1)+len(previous2)):
    data=globals()[f'previous_data_variable{i}']
    data=scaler.transform(data)
    target=globals()[f'previous_target_variable{i}']

    for j in range(len(data)-timestep+1):
        lstm_data_train.append(data[j:j+timestep, :])

    lstm_target_train=np.append(lstm_target_train, target[timestep-1:, :], axis=0)
lstm_data_train=np.array(lstm_data_train)
lstm_target_train=np.delete(lstm_target_train, 0, axis=0)


# Linear
linear = models.Sequential()
linear.add(layers.Input(shape=(10, )))
linear.add(layers.Dense(2))
linear.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
linear.fit(norm_data_train, target_train, batch_size=32, epochs=args.epochs, callbacks=[es])
linear.save('Linear.keras')

# SVR
svr = StandardizedTargetSVR()
svr.partial_fit(norm_data_train, target_train)
pickle.dump(svr, open('SVR.pkl', 'wb'))

# MLP
mlp = models.Sequential()
mlp.add(layers.Input(shape=(10, )))
mlp.add(layers.Dense(2, activation='sigmoid'))
mlp.add(layers.Dense(2))
mlp.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# train model
mlp.fit(norm_data_train, target_train, batch_size=32, epochs=args.epochs, callbacks=[es])
mlp.save('MLP.keras')

# LightGBM
lgbm_x=LGBMRegressor()
lgbm_y=LGBMRegressor()

lgbm_x.fit(norm_data_train, target_train_x)
lgbm_y.fit(norm_data_train, target_train_y)
pickle.dump(lgbm_x, open('LGBM_X.pkl', 'wb'))
pickle.dump(lgbm_y, open('LGBM_Y.pkl', 'wb'))

# LSTM
# define model
lstm = tf.keras.Sequential()
lstm.add(layers.LSTM(10))
lstm.add(layers.Dense(2, activation='sigmoid'))
lstm.add(layers.Dense(2))

lstm.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# train model
lstm.fit(lstm_data_train, lstm_target_train, batch_size=32, epochs=args.epochs, callbacks=[es])
lstm.save(f'LSTM.keras')


# real time learning
# make dataset
data_train = np.zeros((1, 10))
target_train = np.zeros((1, 2))
for i in range(len(additional)):
    data = globals()[f'additional_data_variable{i}']
    if len(data) < test_hours*2:
        continue
    target = globals()[f'additional_target_variable{i}']
    data_train = np.append(data_train, data[:test_hours-3, :], axis=0)
    target_train = np.append(target_train, target[:test_hours-3, :], axis=0)
    
data_train = np.delete(data_train, 0, axis=0)
target_train = np.delete(target_train, 0, axis=0)
target_train_x=np.reshape(target_train[:,0], (-1,))
target_train_y=np.reshape(target_train[:,1], (-1,))

norm_data_train = scaler.transform(data_train)

# make dataset for lstm
lstm_data_train = np.zeros((1, timestep, 10))
lstm_target_train = np.zeros((1,2))
for i in range(len(additional)):
    data=globals()[f'additional_data_variable{i}']
    if len(data) < test_hours*2:
        continue
    data=scaler.transform(data)
    target=globals()[f'additional_target_variable{i}']

    for j in range(test_hours-timestep+1-3):
        lstm_data_train=np.append(lstm_data_train, np.reshape(data[j:j+timestep, :], (1, timestep, 10)), axis=0)

    lstm_target_train=np.append(lstm_target_train, target[timestep-1:test_hours-3, :], axis=0)
lstm_data_train=np.delete(lstm_data_train, 0, axis=0)
lstm_target_train=np.delete(lstm_target_train, 0, axis=0)


# Linear
# model load
linear_inc = models.load_model('Linear.keras')
# incremental(online) learning
linear_inc.fit(norm_data_train, target_train, batch_size=48, epochs=args.incremental_epochs)

# SVR
svr_inc=pickle.load(open('SVR.pkl', 'rb'))
svr_inc.partial_fit(norm_data_train, target_train)

# MLP
# model load
mlp_inc = models.load_model('MLP.keras')
# incremental(online) learning
mlp_inc.fit(norm_data_train, target_train, batch_size=48, epochs=args.incremental_epochs)

# LGBM
lgbm_x_inc=pickle.load(open('LGBM_X.pkl', 'rb'))
lgbm_y_inc=pickle.load(open('LGBM_Y.pkl', 'rb'))
lgbm_x_inc.fit(norm_data_train, target_train_x, init_model=lgbm_x_inc)
lgbm_y_inc.fit(norm_data_train, target_train_y, init_model=lgbm_y_inc)

# LSTM
# model load
lstm_inc = models.load_model('LSTM.keras')
# incremental(online) learning
lstm_inc.fit(lstm_data_train, lstm_target_train, batch_size=48, epochs=args.incremental_epochs)


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

linear_inc_mae=[]
svr_inc_mae=[]
mlp_inc_mae=[]
lgbm_inc_mae=[]
lstm_inc_mae=[]
linear_inc_ncls=[]
svr_inc_ncls=[]
mlp_inc_ncls=[]
lgbm_inc_ncls=[]
lstm_inc_ncls=[]
# test models performance
for k in range(len(additional)):
    # make dataset
    data = globals()[f'additional_data_variable{k}']
    if len(data) < test_hours*2:
        continue
    data_test = data[test_hours-3:test_hours*2-3, :]
    norm_data_test = scaler.transform(data_test)

    # make dataset for lstm
    data=globals()[f'additional_data_variable{k}']
    if len(data) < test_hours*2:
        continue
    data=data[test_hours-timestep+1-3:test_hours*2-3, :]
    lstm_data=scaler.transform(data)

    lstm_data_test=np.zeros((1, timestep, 10))
    for j in range(test_hours):
        lstm_data_test=np.append(lstm_data_test, np.reshape(lstm_data[j:j+timestep, :], (1, timestep, 10)), axis=0)
    lstm_data_test=np.delete(lstm_data_test, 0, axis=0)

    # observation
    filename = add_original + '/' + additional[k] # 바꿀 부분
    file = np.loadtxt(filename, skiprows=1+test_hours, max_rows=test_hours, delimiter=',', usecols=(8,9,10,11)) # 특징확장 데이터와 테스트 개수 맞춤
    start_point = np.loadtxt(filename, skiprows=test_hours, max_rows=1, delimiter=',', usecols=(0, 1))
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
    linear_mae.append(mean_absolute_error(obs_var, pred_var))
    
    linear_loc=start_point
    for j in range(len(pred_var)):
        linear_loc=np.append(linear_loc, np.reshape(linear_loc[j]+pred_var[j], (1,2)), axis=0)
        
    linear_ncls.append(calculate_ncls(obs_loc, linear_loc))

    pred_var=linear_inc.predict(norm_data_test)
    np.savetxt(f'{dir_name}/linear{k}.csv', pred_var, delimiter=',')
    linear_inc_mae.append(mean_absolute_error(obs_var, pred_var))
    
    linear_loc=start_point
    for j in range(len(pred_var)):
        linear_loc=np.append(linear_loc, np.reshape(linear_loc[j]+pred_var[j], (1,2)), axis=0)
        
    linear_inc_ncls.append(calculate_ncls(obs_loc, linear_loc))

    # SVR
    pred_var=svr.predict(norm_data_test)
    svr_mae.append(mean_absolute_error(obs_var, pred_var))
    
    svr_loc=start_point
    for j in range(len(pred_var)):
        svr_loc=np.append(svr_loc, np.reshape(svr_loc[j]+pred_var[j], (1,2)), axis=0)
        
    svr_ncls.append(calculate_ncls(obs_loc, svr_loc))

    pred_var=svr_inc.predict(norm_data_test)
    np.savetxt(f'{dir_name}/svr{k}.csv', pred_var, delimiter=',')
    svr_inc_mae.append(mean_absolute_error(obs_var, pred_var))
    
    svr_loc=start_point
    for j in range(len(pred_var)):
        svr_loc=np.append(svr_loc, np.reshape(svr_loc[j]+pred_var[j], (1,2)), axis=0)
        
    svr_inc_ncls.append(calculate_ncls(obs_loc, svr_loc))

    # MLP
    pred_var=mlp.predict(norm_data_test)
    mlp_mae.append(mean_absolute_error(obs_var, pred_var))
    
    mlp_loc=start_point
    for j in range(len(pred_var)):
        mlp_loc=np.append(mlp_loc, np.reshape(mlp_loc[j]+pred_var[j], (1,2)), axis=0)
        
    mlp_ncls.append(calculate_ncls(obs_loc, mlp_loc))

    pred_var=mlp_inc.predict(norm_data_test)
    np.savetxt(f'{dir_name}/mlp{k}.csv', pred_var, delimiter=',')
    mlp_inc_mae.append(mean_absolute_error(obs_var, pred_var))
    
    mlp_loc=start_point
    for j in range(len(pred_var)):
        mlp_loc=np.append(mlp_loc, np.reshape(mlp_loc[j]+pred_var[j], (1,2)), axis=0)
        
    mlp_inc_ncls.append(calculate_ncls(obs_loc, mlp_loc))

    # LGBM
    pred_x = np.reshape(lgbm_x.predict(norm_data_test), (-1, 1))
    pred_y = np.reshape(lgbm_y.predict(norm_data_test), (-1,1))
    pred_var=np.append(pred_x, pred_y, axis=1)
    lgbm_mae.append(mean_absolute_error(obs_var, pred_var))
    
    lgbm_loc=start_point
    for j in range(len(pred_var)):
        lgbm_loc=np.append(lgbm_loc, np.reshape(lgbm_loc[j]+pred_var[j], (1,2)), axis=0)
        
    lgbm_ncls.append(calculate_ncls(obs_loc, lgbm_loc))

    pred_x = np.reshape(lgbm_x_inc.predict(norm_data_test), (-1, 1))
    pred_y = np.reshape(lgbm_y_inc.predict(norm_data_test), (-1,1))
    pred_var=np.append(pred_x, pred_y, axis=1)
    np.savetxt(f'{dir_name}/lgbm{k}.csv', pred_var, delimiter=',')
    lgbm_inc_mae.append(mean_absolute_error(obs_var, pred_var))
    
    lgbm_loc=start_point
    for j in range(len(pred_var)):
        lgbm_loc=np.append(lgbm_loc, np.reshape(lgbm_loc[j]+pred_var[j], (1,2)), axis=0)
        
    lgbm_inc_ncls.append(calculate_ncls(obs_loc, lgbm_loc))

    # LSTM
    pred_var=lstm.predict(lstm_data_test)
    lstm_mae.append(mean_absolute_error(obs_var, pred_var))
    
    lstm_loc=start_point
    for j in range(len(pred_var)):
        lstm_loc=np.append(lstm_loc, np.reshape(lstm_loc[j]+pred_var[j], (1,2)), axis=0)
        
    lstm_ncls.append(calculate_ncls(obs_loc, lstm_loc))

    pred_var=lstm_inc.predict(lstm_data_test)
    np.savetxt(f'{dir_name}/lstm{k}.csv', pred_var, delimiter=',')
    lstm_inc_mae.append(mean_absolute_error(obs_var, pred_var))
    
    lstm_loc=start_point
    for j in range(len(pred_var)):
        lstm_loc=np.append(lstm_loc, np.reshape(lstm_loc[j]+pred_var[j], (1,2)), axis=0)
        
    lstm_inc_ncls.append(calculate_ncls(obs_loc, lstm_loc))


temp=sys.stdout
sys.stdout = open(args.result_filename or 'expanded_incremental_learning_prediction.txt', 'a') # 바꿀 부분
print('without incremental learning')
print('Test:', add)
print('Baseline:', np.mean(base_mae), np.mean(base_ncls))
print('Linear:', np.mean(linear_mae), np.mean(linear_ncls))
print('SVR:', np.mean(svr_mae), np.mean(svr_ncls))
print('MLP:', np.mean(mlp_mae), np.mean(mlp_ncls))
print('LGBM:', np.mean(lgbm_mae), np.mean(lgbm_ncls))
print('LSTM:', np.mean(lstm_mae), np.mean(lstm_ncls))
print('Average:', np.mean([np.mean(linear_mae), np.mean(svr_mae), np.mean(mlp_mae), np.mean(lgbm_mae), np.mean(lstm_mae)]), np.mean([np.mean(linear_ncls), np.mean(svr_ncls), np.mean(mlp_ncls), np.mean(lgbm_ncls), np.mean(lstm_ncls)]))

print('incremental learning')
print('Baseline:', np.mean(base_mae), np.mean(base_ncls))
print('Linear:', np.mean(linear_inc_mae), np.mean(linear_inc_ncls))
print('SVR:', np.mean(svr_inc_mae), np.mean(svr_inc_ncls))
print('MLP:', np.mean(mlp_inc_mae), np.mean(mlp_inc_ncls))
print('LGBM:', np.mean(lgbm_inc_mae), np.mean(lgbm_inc_ncls))
print('LSTM:', np.mean(lstm_inc_mae), np.mean(lstm_inc_ncls))
print('Average:', np.mean([np.mean(linear_inc_mae), np.mean(svr_inc_mae), np.mean(mlp_inc_mae), np.mean(lgbm_inc_mae), np.mean(lstm_inc_mae)]), np.mean([np.mean(linear_inc_ncls), np.mean(svr_inc_ncls), np.mean(mlp_inc_ncls), np.mean(lgbm_inc_ncls), np.mean(lstm_inc_ncls)]))
sys.stdout.close()
sys.stdout=temp
