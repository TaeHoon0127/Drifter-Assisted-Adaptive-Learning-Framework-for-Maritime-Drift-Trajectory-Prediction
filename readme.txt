1. Principles for Organizing the Released Code

This directory was prepared from the original experimental scripts actually used by the authors.
Wherever possible, the original file names, separation between prediction and correction,
model architectures, training order, data construction using np.append, training/evaluation
intervals, and result-log formats have been preserved.
The code, data, previously trained models, and saved results in the original project were not modified.

Paths and identical execution blocks that had been duplicated for different drifter types
and observation years were replaced with input/output path arguments.
A previously prepared integrated training framework was removed.
Because the 24-hour training code for GA differs from the original incremental-learning
code only in test_hours, it is provided as a short launcher that calls the same original
logic with test_hours=24.

2. Execution Environment: Current Conda 'tensor' Environment

The following values were verified in the actual 'tensor' environment used for the
experiments reported in the manuscript.

  Operating system: Linux x86_64 / WSL2
  Kernel: 5.15.167.4-microsoft-standard-WSL2
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
  WEKA used to verify RBFN: 3.8.7 + RBFNetwork package

The TensorFlow distribution was built for CUDA 12.4. The conda environment includes
cuDNN 9.1.1.17, along with NVIDIA packages installed through pip.
Refer to environment_tensor.yml for the exact conda/pip configuration.

  environment_tensor.yml
    Output of 'conda env export -n tensor --no-builds'; the installation-path
    prefix has been removed.
  environment_tensor_packages.json
    Version/build/channel records for all 199 packages in the current environment.
  runtime_environment.json
    Python/OS/TensorFlow build/Java version information.

To create the environment on another computer:

  conda env create -f environment_tensor.yml
  conda activate tensor
  cd journal_code

3. Paths, Data Formats, and Released Raw-Data Example

data_path and save_path are generic paths to be replaced with the actual data location
and output directory, respectively.

source_a and source_b refer to the two other datasets used for pretraining, whereas
target refers to the dataset used for additional training and evaluation.
For a different observation year or drifter type, supply the corresponding paths to
the same scripts. Files representing the same trajectory must have matching names
in the hourly and expanded directories.
The rules and files used to construct the final merged cohorts from the raw data
for each year must be prepared separately.

The first 10 physical lines of the specified original SVP file were copied verbatim.
These consist of one header line and nine data lines; the values, column order, and
line breaks were not changed.
The file contains time, observed longitude/latitude, modeled longitude/latitude,
and wind/current components at the observed and modeled positions.
It is a partial example intended to illustrate the original data format, not the
complete training dataset.
Because the original dataset-generation code uses only the first two data rows
of each simulation file, a short excerpt from one original file is insufficient
to construct a four-hour history or to perform the full 24-/48-hour evaluation.

hourly CSV: 14 columns without a header. The first numeric row contains the initial
observed longitude/latitude, with zeros in the remaining columns.
Zero-based column indices for subsequent rows:
  0,1: Observed end-position longitude/latitude
  2,3: Modeled end-position longitude/latitude
  4,5,6,7: wind_y, wind_x, current_y, current_x at the starting position
  8,9: Modeled longitude/latitude displacement
  10,11: Observed longitude/latitude displacement
  12,13: Observed displacement minus modeled displacement

expanded CSV: A header beginning with '# ' and 34 numeric columns.
  0..29: 30 input features
  30,31: Prediction targets (observed longitude/latitude displacement)
  32,33: Correction targets (observed displacement minus modeled displacement)
The first numeric row of expanded corresponds to numeric row 4 of hourly.

4. Raw Data -> Hourly Data -> Feature Expansion

1hour_dataset.py
  Purpose: Creates a 14-column hourly CSV for each trajectory by calculating
           positions, environmental components, modeled/observed displacements,
           and correction values from the first two data rows of each simulation CSV.
  The active KAIST computation loop in the original code was adapted to use
  general paths. Repeated commented-out code blocks for SVP/ARGOS were removed.

  python 1hour_dataset.py --data-path data_path/raw --save-path data_path/hourly

To check only the loading of the released raw-data example:

  python 1hour_dataset.py --data-path data_example --save-path save_path/example_hourly

feature_expansion.py
  Purpose: Adds cumulative environmental changes and modeled/observed displacements
           over the preceding 1/2/3 hours to the current six features, producing
           30 input features and four target values.
  The original feature computations and column order have been preserved.

  python feature_expansion.py --data-path data_path/hourly --save-path data_path/expanded

5. Prediction and Correction Using 6/30/10 Features

The eight files described below implement the original leave-one-out (LOO) experiments,
in which each trajectory is held out in turn and the remaining trajectories are used
for training. The models are Linear, SVR, MLP, LightGBM, and LSTM, with their
original model architectures preserved.
Default pretraining settings are epochs=500, batch_size=32, and EarlyStopping
patience=10. The LSTM uses timestep=6, and the six-feature experiments are also
evaluated with the loss of samples caused by the three-hour feature expansion aligned.

original_learning_prediction.py / original_learning_correction.py
  Purpose: Direct prediction of observed displacement / correction of modeled
           displacement using the original six features (hourly columns 4..9).

  python original_learning_prediction.py --data-path data_path/hourly --save-path save_path/original_prediction
  python original_learning_correction.py --data-path data_path/hourly --save-path save_path/original_correction

expanded_learning_prediction.py / expanded_learning_correction.py
  Purpose: Prediction/correction comparison using all 30 expanded features.

  python expanded_learning_prediction.py --data-path data_path/expanded --original-data-path data_path/hourly --save-path save_path/expanded_prediction
  python expanded_learning_correction.py --data-path data_path/expanded --original-data-path data_path/hourly --save-path save_path/expanded_correction

10features_prediction.py / 10features_correction.py
  Purpose: Ten-feature LOO prediction/correction based on the original selected_learning
           implementation (SVR: standardized target; default epsilon=0.1).
  Prediction input indices: 5,8,9,12,13,20,21,26,28,29
  Correction input indices: 4,5,10,11,12,13,19,21,28,29

  python 10features_prediction.py --data-path data_path/expanded --original-data-path data_path/hourly --save-path save_path/10_prediction
  python 10features_correction.py --data-path data_path/expanded --original-data-path data_path/hourly --save-path save_path/10_correction

6. SHAP Feature Selection (30 -> 10 Features)

shap_revision/shap_prediction.py / shap_revision/shap_correction.py
  Purpose: Using all feature indices 0..29, following the original
           shap_prediction.py / shap_correction.py implementations, saves the
           dataset-specific SHAP rankings and top-10 features for LightGBM/LSTM.
shap_revision/shap_runner.py
  Shared implementation. It preserves the original training architecture,
  six-step LSTM, per-axis |SHAP| normalization/averaging, and summation across
  the time axis.

  for cohort in svp_2021 argos_2021 svp_2022 argos_2022 svp_2023 argos_2023 kaist_2023; do
    python shap_revision/shap_prediction.py --data-path data_path/$cohort/expanded --save-path save_path/shap30/$cohort --dataset-label $cohort
    python shap_revision/shap_correction.py --data-path data_path/$cohort/expanded --save-path save_path/shap30/$cohort --dataset-label $cohort
  done

The seven cohort names above are illustrative year/type directory names; replace
them with the actual directories. Each dataset-label must be unique.
Outputs: shap_prediction_<label>.csv / shap_correction_<label>.csv, and TXT files,
figures, and scalers in the original formats.
As in the original implementation, SHAP explanations use the entire training sample.
The LSTM background is a random 10% (at least one sample); default epochs=500.

shap_revision/count_shap_features.py
  Purpose: Automatically aggregates the top-10 features from each model across
           seven datasets. The maximum count is 14 (7 datasets x 2 models),
           while dataset_count is the number of datasets in which a feature is
           selected by at least one model (maximum 7).
  The final 10 features are selected by descending count, then descending
  aggregated SHAP magnitude, and finally ascending feature index.

  python shap_revision/count_shap_features.py --data-path save_path/shap30 --save-path save_path/shap_counts

Outputs: shap_results, prediction_feature_counts.csv, correction_feature_counts.csv,
and selected_features.json.
Feature indices are zero-based, and unselected features are also included.
On rerunning, CSV files are overwritten, whereas TXT files are appended to as in
the original code.
Aggregation does not modify the fixed feature selections in the training scripts.
To check only a subset of results, specify the number with --expected-datasets.

7. Comparison With and Without Incremental Learning and Saving 48-Hour Predictions

expanded_incremental_learning_prediction.py
expanded_incremental_learning_correction.py
  Purpose: Pretrains five models on two other datasets, then performs additional
           training using the initial intervals collected from the target dataset.
           Compares performance over the subsequent 48 hours before and after
           additional training. Post-adaptation prediction CSV files are saved
           as inputs for GA/WMA.

  python expanded_incremental_learning_prediction.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/pred48
  python expanded_incremental_learning_correction.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/cor48

Outputs:
  save_path/pred48/expanded_incremental_learning_prediction.txt
  save_path/cor48/expanded_incremental_learning_correction.txt
  scaler.pkl, Linear.keras, SVR.pkl, MLP.keras,
    LGBM_X.pkl, LGBM_Y.pkl, LSTM.keras under each save_path:
    Used to save the pretrained states, as in the original implementation.
  save_path/predictions/{linear,svr,mlp,lgbm,lstm}<index>.csv under each save_path
  prediction_index.csv and prediction_config.json under each predictions directory

After additional training, the adapted models are evaluated in memory and their
predictions are saved. The saved model files should NOT be interpreted as
post-adaptation models. This saving order is preserved from the original code.
SVR.pkl contains publication_utils.StandardizedTargetSVR with a target scaler;
this module must be on the Python path when loading the file.

The 10 selected inputs used for incremental learning are listed below. They differ
from those in the revision_10features experiment above.
  Prediction: 5,8,9,12,13,20,21,26,28,29
  Correction: 4,5,10,11,12,13,19,21,28,29
The usecols selections, including the two target columns, consist of the above
lists plus 30,31 for prediction or 32,33 for correction, respectively.

With the default test_hours=48, the expanded rows for additional training are
0:45, and the evaluation rows are 45:93.
The hourly file uses skiprows=49 and max_rows=48; the starting position is the
single row at skiprows=48. Since numeric row 0 is the initial position, the
evaluation displacements correspond to numeric rows 49..96.
The original length requirement len(expanded)>=96 is preserved.
Additional training uses neural epochs=300 and batch_size=48. SVR performs one
partial_fit operation, and LightGBM continues training from the existing booster.
The input and target scalers are not refitted.

This experiment is a temporal evaluation that jointly uses the initial intervals
of the target trajectories. It should be distinguished from LOO/CV, which retrains
the models for each individual evaluation trajectory.

8. Generating 24-Hour Predictions for the GA Internal Objective Function

ga_expanded_incremental_learning_prediction.py
ga_expanded_incremental_learning_correction.py
  Purpose: Runs the original incremental-learning logic described above with
           test_hours=24 to produce 24-hour predictions for the GA objective.
           This avoids duplicating the same approximately 450 lines of
           computation code.
  The GA-specific SVR also uses the same 10 selected features as the other
  four models.

  python ga_expanded_incremental_learning_prediction.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/pred24
  python ga_expanded_incremental_learning_correction.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/cor24

Outputs are in pred24/predictions and cor24/predictions.
The TXT logs retain the original GA-specific names:
ga_expanded_incremental_learning_prediction.txt / correction.txt.
The additional-training interval is expanded 0:21, and the objective-function
evaluation interval is 21:45, using hourly numeric rows 25..48.
Therefore, this interval does not overlap with the subsequent 48-hour evaluation.
Since the 24-/48-hour length requirements differ, prepare the two sets of results
for GA using the same trajectory cohort.

9. GA Optimization, Representative Solution, and Statistics

ga_pred_statistical.py
  Purpose: Optimizes the initial GA weights using the preceding 24-hour predictions,
           then evaluates OpenDrift / Uniform average / GA average / Uniform-WMA /
           GA-WMA over the subsequent 48 hours.
ga_pred_statistical_common.py
  Shared implementation of the original GA/statistical procedures. It does not
  need to be executed directly.

  python ga_pred_statistical.py --data-path save_path/pred48/predictions --ga-data-path save_path/pred24/predictions --original-data-path data_path/target/hourly --save-path save_path/ga_pred
  python ga_pred_statistical.py --data-path save_path/cor48/predictions --ga-data-path save_path/cor24/predictions --original-data-path data_path/target/hourly --save-path save_path/ga_cor

Defaults: population=50, generation=100, repeat=30, seed=42,
test_hours=48, ga_test_hours=24, bootstrap=10000.
Among 30 GA runs, the run with the objective-function MAE closest to the median
is selected as the representative solution.
Its weights are used for both GA average and GA-WMA initialization.
An error in comparing different relative fitness values between parents and
offspring was corrected by using a common MAE comparison.
For comparison with the previous behavior, add --legacy-replacement to the
commands above.

The original CSV file names and statistical functions/tests have been preserved:
  ga_run_summary.csv, ga_run_fold_metrics.csv, method_fold_metrics.csv,
  summary_table.csv, result_summary.csv, ga_run_distribution_summary.csv,
  wilcoxon_planned_comparisons.csv, ga_selected_weights.csv, manifest.json,
  boxplots.png
The existing cv/fold column names are retained for result compatibility.
In this incremental-learning/GA evaluation, the statistical unit is a
time-partitioned trajectory, not an independently retrained CV fold.
The original ddof setting and statistical functions are retained; the standard
deviation (SD) was not redefined arbitrarily.
The test is a one-sided paired Wilcoxon test in the direction of GA-WMA improvement,
with Holm correction across the eight comparisons in this run.

10. WMA Initial Weights and Real-Time Updating

realtime_weighted_majority_revision.py
  Purpose: Retains one common computational block from the six repeated
           dataset/task blocks of the original script, and evaluates WMA using
           path and initial-weight arguments.

Using representative GA weights:

  python realtime_weighted_majority_revision.py --data-path save_path/pred48/predictions --original-data-path data_path/target/hourly --ga-weights-path save_path/ga_pred/ga_selected_weights.csv --save-path save_path/wma_pred
  python realtime_weighted_majority_revision.py --data-path save_path/cor48/predictions --original-data-path data_path/target/hourly --ga-weights-path save_path/ga_cor/ga_selected_weights.csv --save-path save_path/wma_cor

Using initial weights all set to 1:

  python realtime_weighted_majority_revision.py --data-path save_path/pred48/predictions --original-data-path data_path/target/hourly --save-path save_path/wma_uniform

Arbitrary initial values may be specified with
--weights w_linear w_svr w_mlp w_lgbm w_lstm.
Weights are initialized separately for each trajectory. After each prediction,
the observed error is used to update the weights for the next time step.
The original weights *= exp(-MAE) rule is evaluated in log space to reduce
underflow.
The result-log file retains its original name, realtime_weighted_majority_ga.txt.
Because WMA uses observations at every time step, this is not an open-loop
48-hour forecast made without intermediate observations.

11. RBFN Comparison Experiment

revision_rbfn_pred.py / revision_rbfn_cor.py
  Purpose: Performs the original six-feature LOO RBFN prediction/correction
           experiment using WEKA RBFRegressor.
revision_rbfn_common.py
  Shared implementation for original ARFF generation, Java execution, result
  parsing, and performance evaluation. It does not need to be run directly.

  python revision_rbfn_pred.py --data-path data_path/hourly --save-path save_path/rbfn_pred --weka-classpath 'data_path/weka.jar:data_path/RBFNetwork.jar'
  python revision_rbfn_cor.py --data-path data_path/hourly --save-path save_path/rbfn_cor --weka-classpath 'data_path/weka.jar:data_path/RBFNetwork.jar'

If Java is not on PATH, add --java-bin <Java executable>.
The JAVA_BIN, WEKA_JAR, and WEKA_CLASSPATH environment variables may also be used.
The classpath separator is ':' on Linux and ';' on Windows.
The original WEKA defaults are retained: -N 2 -R 2 -C 2 -L 1.0E-6,
with an output precision of two decimal places.
Since displacement is measured in degrees, small values may round to zero.
For a precision check, add --decimals 12. To match the distance definition
used in other experiments, add --distance geodesic.
The default NCLS distance uses haversine, as in the original code.

12. SVR Epsilon Ablation and Separate Execution of the Final Configuration

expanded_svr_epsilon_ablation.py
  Purpose: In 30-feature LOO experiments, compares raw-target epsilon=0.1,
           raw-target epsilon=0.01, standardized-target epsilon=0.1,
           and standardized-target epsilon selected by internal validation.
  Internal candidates: 0.01,0.03,0.1,0.3,1.0; at least three trajectories
  are required.
  This is an explicit ablation, so raw-target and alternative-epsilon
  conditions have been retained.

  python expanded_svr_epsilon_ablation.py --data-path data_path/expanded --original-data-path data_path/hourly --save-path save_path/svr_ablation --task both

expanded_incremental_svr_epsilon_ablation.py
  Purpose: SVR-only implementation of the final configuration actually used,
           comparing performance with and without incremental learning over
           48 hours using 10 selected features, epsilon=0.1, and a
           standardized target.
ga_expanded_incremental_svr_epsilon_ablation.py
  Purpose: 24-hour GA-specific execution of the same final 10-feature SVR
           implementation. The old 30-feature code is not used.

  python expanded_incremental_svr_epsilon_ablation.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/svr_only48 --task both
  python ga_expanded_incremental_svr_epsilon_ablation.py --pretrain-data-path data_path/source_a/expanded data_path/source_b/expanded --data-path data_path/target/expanded --original-data-path data_path/target/hourly --save-path save_path/svr_only24 --task both

Outputs: expanded_incremental_svr_epsilon_ablation.txt under each save_path
         (ga_expanded_incremental_svr_epsilon_ablation.txt for the 24-hour GA run),
         plus pred/predictions/svr<index>.csv and cor/predictions/svr<index>.csv.
To run just one task, specify --task prediction or --task correction.
This dedicated implementation is for verifying SVR results; it does not generate
predictions for the other four models.
The same SVR modification has already been applied to the five-model scripts,
so there is no need to run this code separately or manually replace CSV files
in the full pipeline.

publication_utils.py
  Small helper module containing path arguments, recording of file enumeration
  order, standardized-target SVR, mapping from indices to original files,
  GA/WMA input checks, and numerically stable WMA updating.
  It is not a standalone executable script.

13. Interpretation of Results and Scope of Reproducibility

All saved prediction/correction CSV files contain final longitude/latitude
displacements. The correction CSV files already include the modeled displacement;
do not add it again during GA/WMA processing.
Model order: linear, svr, mlp, lgbm, lstm.

MAE is the mean absolute displacement error over time and the longitude/latitude
axes, expressed in degrees.
NCLS is calculated using the original equation from position trajectories obtained
by accumulating displacements from the initial position.
If the observed path length is zero, the undefined NCLS is recorded as NaN.
Consequently, NaN may propagate into some original-format summary logs; check
the affected trajectory in such cases.

The model architectures, default 500/300 epochs, batch sizes, EarlyStopping,
timestep, data splits, original seed settings, and original log/model names have
been preserved. --epochs and --incremental-epochs can be reduced for small-scale
execution checks, but the manuscript settings are used by default.
Full reproduction of the experiments requires the complete datasets beyond the
released example, as well as the exact cohort construction and file ordering.
