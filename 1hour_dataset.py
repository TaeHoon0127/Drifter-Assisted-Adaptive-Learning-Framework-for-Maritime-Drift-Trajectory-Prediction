import numpy as np
import os
import warnings
from publication_utils import arguments, output_directory, csv_files

args = arguments("dataset")
output_directory(args)

#load csv files
dir = sorted(p for p in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, p)))
stride, offset = args.directory_stride, args.directory_offset
if stride < 1 or not 0 <= offset < stride:
    raise ValueError('Require stride >= 1 and 0 <= offset < stride')
if not dir:
    raise ValueError('No raw trajectory subdirectories')
dir.sort()


for i in range(len(dir)//stride):
    filelist=csv_files(os.path.join(args.data_path, dir[stride*i+offset]))
    if not filelist:
        raise ValueError("Empty trajectory directory")
    previous_endpoint = None
    discontinuities = 0
    filelist.sort()
    dataset_hour=np.zeros((1, 14))
    for j in range(len(filelist)):
        if filelist[j] == 'desktop.ini':
            continue
        filename=os.path.join(args.data_path, dir[stride*i+offset], filelist[j])
        file = np.loadtxt(filename, skiprows=1, max_rows=2, delimiter=',', usecols=(1, 2, 3, 4, 9, 10, 11, 12))

        if j==0:
            start_point=np.reshape(file[0, :2], (1,2))
            start_point=np.append(start_point, np.zeros((1,12)), axis=1)
            dataset_hour=np.append(dataset_hour, start_point, axis=0)

        location=np.reshape(file[1, :4], (1,4)) # mae 계산용 두 번째 줄 obs, model 위치
        weather=np.reshape(file[0, 4:], (1,4)) # 시작위치 바람, 해류
        model=file[:2, 2:4]
        model_var=np.reshape(model[1]-model[0], (1,2)) # model 변화량
        obs=file[:2, :2]
        obs_var=np.reshape(obs[1]-obs[0], (1,2)) # obs 변화량
        dif=obs_var-model_var # 변화량 간 차이

        dataset=location
        dataset=np.append(dataset, weather, axis=1)
        dataset=np.append(dataset, model_var, axis=1)
        dataset=np.append(dataset, obs_var, axis=1)
        dataset=np.append(dataset, dif, axis=1)
        has_nan = not np.isfinite(dataset).all()
        if has_nan:
            if args.invalid_rows != 'skip':
                raise ValueError(f'Non-finite dataset row: {filename}; use --invalid-rows skip for original behavior')
            print(f'skip invalid sample: {filename} (source file unchanged; hourly continuity may be lost)')
            continue
        if previous_endpoint is not None and not np.allclose(file[0, :2], previous_endpoint, rtol=0, atol=1e-7):
            discontinuities += 1
            if args.strict_continuity:
                raise ValueError(f'Nonmatching adjacent observations before {filename}')
        previous_endpoint = file[1, :2]
        
        dataset_hour=np.append(dataset_hour, dataset, axis=0)

    dataset_hour=np.delete(dataset_hour, 0, axis=0)
    if discontinuities:
        warnings.warn(f'{dir[stride*i+offset]}: {discontinuities} nonmatching adjacent observation positions; original displacement formula preserved')
    np.savetxt(os.path.join(args.save_path, dir[stride*i+offset] + '.csv'), dataset_hour, delimiter=',')
