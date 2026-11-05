import numpy as np
import os

from publication_utils import arguments, csv_files

args = arguments()
dataset_name = args.data_path
#load csv files
filelist = csv_files(dataset_name, sort=True)


for i in range(len(filelist)):
    filename=dataset_name + '/' + filelist[i]
    file = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13), ndmin=2)
    if len(file) < 4:
        raise ValueError('Feature expansion requires at least four hourly displacement rows')

    dataset_30features=np.zeros((1,34))
    for j in range(len(file)-3):
        dataset=np.reshape(file[j+3, :6], (1,6)) # 바람, 해류, model변화량 데이터셋(6개)

        # var1=현재-1시간전, var2=1시간전-2시간전, var3=2시간전-3시간전
        # 바람,해류 변화량
        wincur_var1=file[j+3, :4]-file[j+2, :4]
        wincur_var2=file[j+2, :4]-file[j+1, :4]
        wincur_var3=file[j+1, :4]-file[j, :4]
        #모델, obs 변화량
        modobs_var1=file[j+2, 4:8]
        modobs_var2=file[j+1, 4:8]
        modobs_var3=file[j, 4:8]

        dataset_var1=np.append(wincur_var1, modobs_var1)
        dataset_var1=np.reshape(dataset_var1, (1,8)) # 바람,해류,모델,obs var1 데이터셋(8개)

        wincur_var12=wincur_var1+wincur_var2
        modobs_var12=modobs_var1+modobs_var2
        dataset_var2=np.append(wincur_var12, modobs_var12)
        dataset_var2=np.reshape(dataset_var2, (1,8)) # 바람,해류,모델,obs var1+var2 데이터셋(8개)

        wincur_var123=wincur_var1+wincur_var2+wincur_var3
        modobs_var123=modobs_var1+modobs_var2+modobs_var3
        dataset_var3=np.append(wincur_var123, modobs_var123)
        dataset_var3=np.reshape(dataset_var3, (1,8)) # 바람,해류,모델,obs var1+var2+var3 데이터셋(8개)

        dataset_target=np.reshape(file[j+3, 6:10], (1,4)) # target obs_var, obs_var-model_var 데이터셋(4개)

        # append all features
        dataset = np.append(dataset, dataset_var1, axis=1)
        dataset = np.append(dataset, dataset_var2, axis=1)
        dataset = np.append(dataset, dataset_var3, axis=1)
        dataset = np.append(dataset, dataset_target, axis=1)

        # add to dataset_30features
        dataset_30features=np.append(dataset_30features, dataset, axis=0)

    # save file
    dataset_30features=np.delete(dataset_30features, 0, axis=0)
    np.savetxt(os.path.join(args.save_path, filelist[i]), dataset_30features, delimiter=',', header='wind_y, wind_x, current_y, current_x, model_lon_var, model_lat_var, wind_y_var1, wind_x_var1, current_y_var1, current_x_var1, model_lon_var1, model_lat_var1, obs_lon_var1, obs_lat_var1, wind_y_var2, wind_x_var2, current_y_var2, current_x_var2, model_lon_var2, model_lat_var2, obs_lon_var2, obs_lat_var2, wind_y_var3, wind_x_var3, current_y_var3, current_x_var3, model_lon_var3, model_lat_var3, obs_lon_var3, obs_lat_var3, prediction_lon, prediction_lat, correction_lon, correction_lat')
