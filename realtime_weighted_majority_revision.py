import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import copy
import sys
from geopy.distance import geodesic

from publication_utils import (arguments, output_directory, initial_weights,
                               prediction_pairs, stable_weighted_majority_average)

args = arguments("wma")
if args.original_data_path is None:
    raise ValueError("--original-data-path is required")
starting_weights = initial_weights(args)
output_directory(args)


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


temp=sys.stdout
sys.stdout = open('realtime_weighted_majority_ga.txt', 'a') # 바꿀 부분
    
    
# parameters
test_dir = args.data_path
add_original = args.original_data_path
pairs = prediction_pairs(test_dir, add_original, args.test_hours)
additional = [filename for _, filename in pairs]

test_hours = args.test_hours

ensemble_mae=[]
ensemble_ncls=[]
for k in range(len(additional)):
#     weights = copy.deepcopy(initial_weights[i])
    # weights = [1,1,1,1,1]
    weights = starting_weights.copy()
    prediction_index = pairs[k][0]
    
    # observation
    filename = add_original + '/' + additional[k] # 바꿀 부분
    file = np.loadtxt(filename, skiprows=1+test_hours, max_rows=test_hours, delimiter=',', usecols=(8,9,10,11)) # 특징확장 데이터와 테스트 개수 맞춤
    start_point = np.loadtxt(filename, skiprows=test_hours, max_rows=1, delimiter=',', usecols=(0, 1))
    start_point = np.reshape(start_point, (1,2))
    obs_var = file[:, 2:]

    obs_loc=start_point
    for j in range(len(obs_var)):
        obs_loc=np.append(obs_loc, np.reshape(obs_loc[j]+obs_var[j], (1,2)), axis=0)
    
    # Ensemble
    linear = np.loadtxt(test_dir + '/' + f'linear{prediction_index}.csv', delimiter=',')
    svr = np.loadtxt(test_dir + '/' + f'svr{prediction_index}.csv', delimiter=',')
    mlp = np.loadtxt(test_dir + '/' + f'mlp{prediction_index}.csv', delimiter=',')
    lgbm = np.loadtxt(test_dir + '/' + f'lgbm{prediction_index}.csv', delimiter=',')
    lstm = np.loadtxt(test_dir + '/' + f'lstm{prediction_index}.csv', delimiter=',')
    
    ensemble_var = stable_weighted_majority_average(
        np.stack((linear, svr, mlp, lgbm, lstm)), obs_var, weights)

    ensemble_mae.append(mean_absolute_error(obs_var, ensemble_var))

    ensemble_loc=start_point
    for j in range(len(ensemble_var)):
        ensemble_loc=np.append(ensemble_loc, np.reshape(ensemble_loc[j]+ensemble_var[j], (1,2)), axis=0)
        
    ensemble_ncls.append(calculate_ncls(obs_loc, ensemble_loc))
print('Test: ', test_dir)
method = 'GA+WMA' if args.ga_weights_path else ('Custom+WMA' if args.weights else 'Uniform+WMA')
print(method + ': ', np.mean(ensemble_mae), np.mean(ensemble_ncls))
# print('Uniform+WMA: ', np.mean(ensemble_mae), np.mean(ensemble_ncls))

sys.stdout.close()
sys.stdout = temp
