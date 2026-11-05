실험 코드 공개 및 실행 안내
작성일: 2026-11-05

1. 공개 코드의 구성 원칙

이 디렉터리는 저자가 실제 사용한 원본 실험 스크립트를 기반으로 작성했다.
원본의 파일명, prediction/correction 구분, 모델 구조, 학습 순서, np.append를
이용한 데이터 구성, 학습/평가 구간 및 결과 로그 형식을 가능한 한 유지했다.
원본 프로젝트의 코드, 데이터, 기존 학습 모델 및 저장 결과는 변경하지 않았다.

데이터 종류/관측 연도에 따라 복사해 두었던 경로 및 동일한 실행 블록을
입력/출력 경로 인수로 대체했다. 이전에 작성했던 통합 학습 프레임워크는 제거했다.
GA용 24시간 학습 코드는 원래 증분학습 코드와 test_hours만 달랐으므로,
같은 원본 로직을 test_hours=24로 호출하는 짧은 실행 파일로 제공한다.

2. 실행 환경: 현재 conda tensor 환경

아래는 논문의 실험이 수행된 실제 tensor 환경에서 확인한 값이다.

  운영체제: Linux x86_64 / WSL2
  커널: 5.15.167.4-microsoft-standard-WSL2
  glibc: 2.35
  Python: 3.12.9 (Anaconda, GCC 11.2.0)
  NumPy: 2.0.1
  scikit-learn: 1.6.1
  TensorFlow: 2.18.1
  Keras: 3.9.2
  LightGBM: 4.6.0
  SHAP: 0.47.2
  geopy: 2.4.1
  SciPy: 1.15.2
  Matplotlib: 3.10.6
  joblib: 1.4.2
  pandas: 2.3.1
  Java: OpenJDK 11.0.1-internal
  RBFN 확인에 사용한 WEKA: 3.8.7 + RBFNetwork 패키지

TensorFlow는 CUDA 12.4를 대상으로 빌드된 배포판이다. conda 환경에는
cuDNN 9.1.1.17이 있으며 pip로 설치된 NVIDIA 패키지도 함께 존재한다.
정확한 conda/pip 구성은 environment_tensor.yml을 기준으로 확인한다.

  environment_tensor.yml
    conda env export -n tensor --no-builds 결과. 설치 경로 prefix는 제거했다.
  environment_tensor_packages.json
    현재 전체 199개 패키지의 버전/build/channel 기록.
  runtime_environment.json
    Python/OS/TensorFlow 빌드/Java 버전 기록.

다른 컴퓨터에 환경을 생성하는 경우:

  conda env create -f environment_tensor.yml
  conda activate tensor
  cd journal_code

3. 경로, 데이터 형식 및 공개 원본 예시

data_path와 save_path는 실제 데이터 위치와 결과 저장 위치로 바꾸는 일반 경로이다.

source_a/source_b는 사전학습에 사용한 다른 두 데이터셋이고 target은 추가 학습
및 평가 대상 데이터셋이다. 관측 연도/종류가 바뀌면 동일 스크립트에 해당 경로를 준다.
hourly와 expanded의 같은 궤적은 파일명이 일치해야 한다.
연도별 원자료를 최종 merged 코호트로 구성하는 규칙/파일은 별도로 준비해야 한다.


사용자가 지정한 SVP 원본 파일의 첫 10개 물리적 줄을 그대로 복사했다.
헤더 1줄과 데이터 9줄이며 값, 열 순서 및 개행을 수정하지 않았다.
time, 관측 경위도, 모델 경위도, 관측/모델 위치의 바람 및 해류 성분이 포함된다.
이 파일은 원본 형식 설명을 위한 일부 예시이며 전체 학습 데이터가 아니다.
원본 데이터셋 생성 코드는 시뮬레이션 파일마다 앞 두 데이터 행만 사용하므로,
원본 파일 하나의 일부만으로 4시간 이력이나 24/48시간 평가 전체를 만들 수 없다.

hourly CSV: 헤더 없이 14열. 첫 숫자 행은 초기 관측 경위도와 나머지 0.
이후 행의 0-based 열 인덱스:
  0,1: 관측 끝 위치 경도/위도
  2,3: 모델 끝 위치 경도/위도
  4,5,6,7: 시작 위치의 wind_y, wind_x, current_y, current_x
  8,9: 모델 경도/위도 변위
  10,11: 관측 경도/위도 변위
  12,13: 관측 변위 - 모델 변위

expanded CSV: '# '로 시작하는 헤더와 숫자 34열.
  0..29: 입력 특징 30개
  30,31: prediction 정답, 관측 경도/위도 변위
  32,33: correction 정답, 관측 변위 - 모델 변위
expanded의 첫 숫자 행은 hourly의 숫자 행 4에 대응한다.

4. 원본 데이터 -> 시간 단위 데이터 -> 특징확장

1hour_dataset.py
  기능: 각 시뮬레이션 CSV의 첫 두 데이터 행에서 위치, 환경 성분, 모델/관측
        변위와 보정량을 계산해 궤적별 14열 hourly CSV를 만든다.
  원본의 활성 KAIST 계산 루프를 일반 경로로 바꾸었다.
  SVP/ARGOS용으로 반복된 주석 코드 블록은 제거했다.

  python 1hour_dataset.py --data-path data_path/raw --save-path data_path/hourly

공개 예시의 원본 로딩만 확인:

  python 1hour_dataset.py --data-path data_example --save-path save_path/example_hourly

feature_expansion.py
  기능: 현재 6특징에 과거 1/2/3시간 누적 환경 변화와 모델/관측 변위를 붙여
        30개 입력과 4개 정답을 만든다. 원본 특징 계산/열 순서를 유지했다.

  python feature_expansion.py --data-path data_path/hourly --save-path data_path/expanded

5. 6/30/10특징 prediction 및 correction

아래 8개 파일은 각 궤적 하나를 제외한 나머지로 학습하는 원본 LOO 실험이다.
모델은 Linear, SVR, MLP, LightGBM, LSTM이며 원본의 모델 구조를 유지한다.
기본 사전학습 epochs=500, batch_size=32, EarlyStopping patience=10이다.
LSTM timestep=6이며 여섯 특징 실험도 3시간 특징확장 손실을 맞춰 평가한다.

original_learning_prediction.py / original_learning_correction.py
  기능: 원래 6특징(hourly 4..9)으로 관측 변위 직접 예측 / 모델 변위 보정.

  python original_learning_prediction.py --data-path data_path/hourly --save-path save_path/original_prediction
  python original_learning_correction.py --data-path data_path/hourly --save-path save_path/original_correction

expanded_learning_prediction.py / expanded_learning_correction.py
  기능: 확장된 30특징 모두를 사용한 예측 / 보정 비교.

  python expanded_learning_prediction.py --data-path data_path/expanded --original-data-path data_path/hourly --save-path save_path/expanded_prediction
  python expanded_learning_correction.py --data-path data_path/expanded --original-data-path data_path/hourly --save-path save_path/expanded_correction

10features_prediction.py / 10features_correction.py
  기능: selected_learning 원본 기준 10특징 LOO 예측 / 보정 (SVR: target standardize, 기본 epsilon=0.1).
  prediction 입력: 5,8,9,12,13,20,21,26,28,29
  correction 입력: 4,5,10,11,12,13,19,21,28,29

  python 10features_prediction.py --data-path data_path/expanded --original-data-path data_path/hourly --save-path save_path/10_prediction
  python 10features_correction.py --data-path data_path/expanded --original-data-path data_path/hourly --save-path save_path/10_correction

6. SHAP feature selection (30개 -> 10개)

shap_revision/shap_prediction.py / shap_revision/shap_correction.py
  기능: shap_prediction.py / shap_correction.py 원본 기준으로 0~29번 특징 전체를
        사용해 데이터셋별 LightGBM/LSTM의 SHAP 순위 및 각 모델 top-10을 저장한다.
shap_revision/shap_runner.py
  공통 구현. 원본 학습 구조, 6시점 LSTM, 축별 |SHAP| 정규화/평균 및 시간축 합을 유지한다.

  for cohort in svp_2021 argos_2021 svp_2022 argos_2022 svp_2023 argos_2023 kaist_2023; do
    python shap_revision/shap_prediction.py --data-path data_path/$cohort/expanded --save-path save_path/shap30/$cohort --dataset-label $cohort
    python shap_revision/shap_correction.py --data-path data_path/$cohort/expanded --save-path save_path/shap30/$cohort --dataset-label $cohort
  done
위 7개 이름은 연도/종류별 경로 예시이며 실제 폴더에 맞게 바꾼다. dataset-label은 서로 달라야 한다.
출력: shap_prediction_<label>.csv / shap_correction_<label>.csv 및 원본 형식 TXT/그림/scaler.
원본처럼 전체 학습 표본을 설명한다. LSTM background는 무작위 10%(최소 1개), 기본 epochs=500.

shap_revision/count_shap_features.py
  기능: 7개 데이터셋의 각 모델 top-10을 자동 집계한다. count는 최대 14회(7개 x 2모델),
        dataset_count는 한 모델 이상에서 선택된 데이터셋 수(최대 7개)이다.
  count 내림차순, 선택된 SHAP 합 내림차순, 특징 번호 오름차순으로 최종 10개를 선택한다.

  python shap_revision/count_shap_features.py --data-path save_path/shap30 --save-path save_path/shap_counts

출력: shap_results, prediction_feature_counts.csv, correction_feature_counts.csv, selected_features.json.
특징 번호는 0 기준이며 미선택 특징도 포함한다. 재실행 시 CSV는 덮어쓰고 TXT는 원본처럼 누적한다.
집계는 학습 코드의 고정 특징을 바꾸지 않는다. 일부 집계만 확인할 때는 --expected-datasets 개수를 지정한다.

7. Incremental learning 유무 비교 및 48시간 예측 저장

expanded_incremental_learning_prediction.py
expanded_incremental_learning_correction.py
  기능: 다른 두 데이터셋으로 5개 모델을 사전학습하고, 대상 데이터셋의 앞 구간을
        모아서 추가 학습한다. 추가 학습 전/후의 이후 48시간 성능을 비교한다.
        추가 학습 후 예측 CSV는 GA/WMA 입력으로 저장한다.

  python expanded_incremental_learning_prediction.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/pred48
  python expanded_incremental_learning_correction.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/cor48

출력:
  save_path/pred48/expanded_incremental_learning_prediction.txt
  save_path/cor48/expanded_incremental_learning_correction.txt
  각 save_path의 scaler.pkl, Linear.keras, SVR.pkl, MLP.keras,
    LGBM_X.pkl, LGBM_Y.pkl, LSTM.keras: 원본처럼 사전학습 상태 저장에 사용.
  각 save_path/predictions/{linear,svr,mlp,lgbm,lstm}<index>.csv
  각 predictions/prediction_index.csv 및 prediction_config.json

추가 학습 후 모델은 메모리에서 평가하고 예측을 저장한다. 저장된 모델 파일이
추가 학습 후 모델이라고 해석하면 안 된다. 이 저장 순서도 원본을 유지했다.
SVR.pkl에는 타깃 scaler가 포함된 publication_utils.StandardizedTargetSVR가
저장되므로 로딩 시 이 모듈을 Python 경로에 포함해야 한다.

증분학습의 선택 입력은 아래 10개로, 위 revision_10features 실험과 다르다.
  prediction: 5,8,9,12,13,20,21,26,28,29
  correction: 4,5,10,11,12,13,19,21,28,29
정답 2개를 포함한 usecols는 각각 위 목록 + 30,31 / 32,33이다.

기본 test_hours=48일 때 expanded 추가 학습 0:45, 평가 45:93이다.
hourly 파일은 skiprows=49, max_rows=48, 시작점은 skiprows=48의 한 행이다.
숫자 행 0이 초기 위치이므로 평가 변위는 숫자 행 49..96이다.
길이 조건 len(expanded)>=96도 원본과 동일하다.
추가 학습은 neural epochs=300, batch_size=48이고 SVR은 partial_fit 한 번,
LightGBM은 기존 booster에서 이어서 학습한다. 입력/타깃 scaler는 재학습하지 않는다.

이 실험은 대상 궤적들의 앞 구간을 함께 사용한 temporal evaluation이다.
각 평가 궤적마다 모델을 다시 학습하는 LOO/CV와 구분해야 한다.

8. GA 내부 목적함수용 24시간 예측

ga_expanded_incremental_learning_prediction.py
ga_expanded_incremental_learning_correction.py
  기능: 앞의 원본 증분학습 로직을 test_hours=24로 실행해 GA 목적함수용
        24시간 예측을 만든다. 450줄의 동일한 계산 코드를 중복 복사하지 않는다.
  GA용 SVR도 다른 4개 모델과 동일한 선택 10특징을 사용한다.

  python ga_expanded_incremental_learning_prediction.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/pred24
  python ga_expanded_incremental_learning_correction.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/cor24

출력은 pred24/predictions 및 cor24/predictions이다. txt 로그는 원래 GA용 이름인
ga_expanded_incremental_learning_prediction.txt / correction.txt를 유지한다.
추가 학습 구간은 expanded 0:21, 목적함수 평가 구간은 21:45이고,
hourly 숫자 행 25..48을 사용한다. 따라서 이후 48시간 평가 구간과 겹치지 않는다.
24/48시간 길이 조건이 다르므로 GA에 줄 두 결과는 같은 궤적 코호트로 준비해야 한다.

9. GA 최적화, 대표 해 및 통계

ga_pred_statistical.py
  기능: 앞선 24시간 예측에서 GA 초기 가중치를 최적화한 뒤, 이후 48시간에서
        OpenDrift / Uniform average / GA average / Uniform-WMA / GA-WMA를 평가한다.
ga_pred_statistical_common.py
  원본 GA/통계 공통 구현. 직접 실행할 필요는 없다.

  python ga_pred_statistical.py --data-path save_path/pred48/predictions --ga-data-path save_path/pred24/predictions --original-data-path data_path/target/hourly --save-path save_path/ga_pred
  python ga_pred_statistical.py --data-path save_path/cor48/predictions --ga-data-path save_path/cor24/predictions --original-data-path data_path/target/hourly --save-path save_path/ga_cor

기본값: population=50, generation=100, repeat=30, seed=42,
test_hours=48, ga_test_hours=24, bootstrap=10000.
GA는 30회 실행 목적함수 MAE 중앙값에 가장 가까운 실행을 대표 해로 선택한다.
해당 가중치는 GA average와 GA-WMA 초기화에 함께 사용한다.
부모/자식의 서로 다른 상대 fitness를 비교하던 오류는 공통 MAE 비교로 수정했다.
과거 동작 대조가 필요하면 위 명령에 --legacy-replacement를 추가한다.

원본 CSV 이름과 통계 함수/검정 구성을 유지했다.
  ga_run_summary.csv, ga_run_fold_metrics.csv, method_fold_metrics.csv,
  summary_table.csv, result_summary.csv, ga_run_distribution_summary.csv,
  wilcoxon_planned_comparisons.csv, ga_selected_weights.csv, manifest.json,
  boxplots.png
기존 cv/fold 열 이름은 결과 호환을 위해 남겼다. 이 incremental/GA 평가에서
해당 통계 단위는 독립 재학습 CV fold가 아니라 시간 분할된 궤적이다.
원래 ddof 설정과 통계 함수를 유지했으므로 SD를 임의로 재정의하지 않았다.
검정은 GA-WMA의 개선 방향 단측 paired Wilcoxon이며, 이 실행의 8개 비교에 Holm 보정한다.

10. WMA 초기 가중치 및 실시간 갱신

realtime_weighted_majority_revision.py
  기능: 원본의 6개 데이터셋/task 반복 블록 중 같은 계산 블록 하나를 유지하고,
        경로와 초기 가중치를 인수로 받아 WMA를 평가한다.

GA 대표 가중치를 사용하는 경우:

  python realtime_weighted_majority_revision.py --data-path save_path/pred48/predictions --original-data-path data_path/target/hourly --ga-weights-path save_path/ga_pred/ga_selected_weights.csv --save-path save_path/wma_pred
  python realtime_weighted_majority_revision.py --data-path save_path/cor48/predictions --original-data-path data_path/target/hourly --ga-weights-path save_path/ga_cor/ga_selected_weights.csv --save-path save_path/wma_cor

처음 가중치를 모두 1로 사용하는 경우:

  python realtime_weighted_majority_revision.py --data-path save_path/pred48/predictions --original-data-path data_path/target/hourly --save-path save_path/wma_uniform

임의의 초기값은 --weights w_linear w_svr w_mlp w_lgbm w_lstm으로 지정한다.
각 궤적에서 초기화하고, 당해 예측 후 관측 오차로 다음 시간의 가중치를 갱신한다.
원래 weights *= exp(-MAE)와 동일한 규칙을 log 좌표로 계산해 underflow를 줄였다.
결과 로그 파일은 원본 이름 realtime_weighted_majority_ga.txt를 유지한다.
WMA는 매 시간 관측을 사용하므로 관측 없이 48시간을 내다보는 open-loop 예보가 아니다.

11. RBFN 비교 실험

revision_rbfn_pred.py / revision_rbfn_cor.py
  기능: 원본의 6특징 LOO RBFN 예측 / 보정 실험을 WEKA RBFRegressor로 수행한다.
revision_rbfn_common.py
  원본 ARFF 생성, Java 실행, 결과 파싱 및 성능 평가 구현. 직접 실행하지 않는다.

  python revision_rbfn_pred.py --data-path data_path/hourly --save-path save_path/rbfn_pred --weka-classpath 'data_path/weka.jar:data_path/RBFNetwork.jar'
  python revision_rbfn_cor.py --data-path data_path/hourly --save-path save_path/rbfn_cor --weka-classpath 'data_path/weka.jar:data_path/RBFNetwork.jar'

Java가 PATH에 없다면 --java-bin <Java 실행 파일>을 추가한다.
JAVA_BIN, WEKA_JAR, WEKA_CLASSPATH 환경변수도 사용할 수 있다.
Linux classpath 구분자는 ':'이고 Windows는 ';'이다.
WEKA 옵션 -N 2 -R 2 -C 2 -L 1.0E-6, 출력 정밀도 2자리의 원래 기본값을 유지했다.
변위는 degree 단위이므로 작은 값이 0으로 반올림될 수 있다.
정밀도 확인 실험은 --decimals 12, 다른 실험과 거리 정의를 맞추려면
--distance geodesic을 추가한다. 기본 NCLS 거리는 원본처럼 haversine이다.

12. SVR epsilon ablation 및 최종 설정만 별도 실행

expanded_svr_epsilon_ablation.py
  기능: 30특징 LOO에서 raw target epsilon 0.1 / raw target epsilon 0.01 /
        standardized target epsilon 0.1 / standardized target + 내부 검증 epsilon을 비교.
  내부 후보는 0.01,0.03,0.1,0.3,1.0이며 최소 세 궤적이 필요하다.
  명시적인 ablation이므로 raw target/다른 epsilon 조건을 남겼다.

  python expanded_svr_epsilon_ablation.py --data-path data_path/expanded --original-data-path data_path/hourly --save-path save_path/svr_ablation --task both

expanded_incremental_svr_epsilon_ablation.py
  기능: 실제 사용한 최종 SVR 전용 코드. 선택 10특징, epsilon=0.1,
        standardized target으로 incremental 유무를 48시간에서 비교한다.
ga_expanded_incremental_svr_epsilon_ablation.py
  기능: 같은 최종 10특징 SVR 코드의 GA용 24시간 실행. 과거 30특징 코드는 사용하지 않는다.

  python expanded_incremental_svr_epsilon_ablation.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/svr_only48 --task both
  python ga_expanded_incremental_svr_epsilon_ablation.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/svr_only24 --task both

출력: 각 save_path의 expanded_incremental_svr_epsilon_ablation.txt
      (24시간 GA용은 ga_expanded_incremental_svr_epsilon_ablation.txt)와
      pred/predictions/svr<index>.csv, cor/predictions/svr<index>.csv.
--task prediction 또는 --task correction으로 하나만 실행할 수도 있다.
이 전용 코드는 SVR 결과 검증용이며 나머지 네 모델의 예측은 만들지 않는다.
앞의 five-model 코드에도 이미 같은 SVR 수정을 적용했으므로 전체 파이프라인에서
이 코드를 따로 실행하거나 CSV를 수동 교체할 필요는 없다.

publication_utils.py
  경로 인수, 파일 열거 순서 기록, 표준화 타깃 SVR, 번호-원본 매핑,
  GA/WMA 입력 검사, 안정적인 WMA 갱신만 담은 작은 보조 모듈이다.
  독립 실행용 프로그램이 아니다.

13. 결과의 의미와 재현성 범위

prediction/correction의 저장 예측 CSV는 모두 최종 경도/위도 변위이다.
correction CSV에는 모델 변위를 이미 더했으므로 GA/WMA에서 다시 더하지 않는다.
모델 순서는 linear, svr, mlp, lgbm, lstm이다.

MAE는 시간과 경도/위도 두 축에 대한 평균 절대 변위 오차이며 degree 단위다.
NCLS는 시작점에서 변위를 누적한 위치 궤적을 이용해 원본 식으로 계산한다.
관측 경로 길이가 0이면 정의되지 않은 NCLS를 NaN으로 기록한다.
이때 일부 원본 형식 요약 로그에는 NaN이 전파될 수 있으므로 해당 궤적을 확인해야 한다.

모델 구조, 기본 500/300 epochs, batch size, EarlyStopping, timestep, 분할,
원본 seed 설정, 기존 로그/모델 이름은 유지했다. --epochs와 --incremental-epochs는
작은 실행 검증용으로 줄일 수 있으나 논문 설정은 기본값을 사용한다.
전체 실험의 재현에는 예시 외 전체 데이터, 정확한 코호트 구성과 파일 순서가 필요하다.