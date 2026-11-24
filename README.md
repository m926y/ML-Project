# CAN-Based Lane Keeping Assist Anomaly Detection

This machine learning project classifies the health of a Lane Keeping Assist (LKA) system at the frame level using CAN data from a Tesla AP3 Model 3 equipped with OpenPilot. The mean signed `LKA_error` in each paired CAN window defines the label. Values below 0.25 are normal (class 1), and values at or above 0.25 are anomalous (class 2). Model inputs use vehicle and lane-perception signals. The target variable is reserved for labels and error analysis.

The [Preliminary Report](Preliminary%20Report/Preliminary%20Report.docx) documents the feature construction, model training, validation and limitations.

## Setup and data

The pipeline was verified with Python 3.14.3. From the repository root, install the pinned dependencies in a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Download `TESLA_AP3_MODEL3_v1.rar` from the [course dataset link](https://cloud.tsinghua.edu.cn/f/28927d6840524ec98ed7/?dl=1). The archive is about 543 MB and expands to 2.93 GB, so both forms are kept outside Git. Run the following command with 7-Zip or extract manually to the same directory.

```powershell
7z x -odata TESLA_AP3_MODEL3_v1.rar
```

The expected layout is `data/TESLA_AP3_MODEL3_v1/{Train,Test}/record*/`, with `can_*.csv` and matching `frame_*.jpg` files in each record.

## Reproduce the experiments

Run the scripts in order from the repository root. Each script derives its default paths from the project directory.

```powershell
.\.venv\Scripts\python preprocess.py
.\.venv\Scripts\python experiment.py
.\.venv\Scripts\python extension.py
.\.venv\Scripts\python inspect_results.py
```

| Script | Main outputs |
| --- | --- |
| `preprocess.py` | Frame pairing audit and CAN feature table in `results/` |
| `experiment.py` | Grouped fold assignments, logistic and random-forest comparisons, predictions and fitted models |
| `extension.py` | Ratio and concentration features, random-forest and boosting comparisons, and paired uncertainty intervals in `results/extension/` |
| `inspect_results.py` | Error breakdown and camera/CAN examples for the base random-forest experiment |

The scripts preserve the supplied Train/Test split. The 2,409 usable training frames are grouped into five stratified folds by clip and timestamp overlap. The 473 usable test frames are scored with hyperparameters and alert thresholds chosen within Train. Generated tables, plots and fitted models stay in `results/`.

## Selected results

| Model and feature set | Mean validation AP | Test AP | Test F1 |
| --- | ---: | ---: | ---: |
| Constant prior | — | 0.207 | 0.000 |
| Logistic regression, C | 0.599 | 0.300 | 0.172 |
| Random forest, C | 0.804 | 0.401 | 0.374 |
| Random forest, C + ratios | **0.818** | **0.421** | 0.457 |
| XGBoost, C + ratios + concentration | 0.800 | 0.359 | 0.418 |
| HistGradientBoosting, C | 0.786 | 0.302 | **0.466** |

Set C contains signal means, torque variability, the lane-confidence gap and the mean absolute curvature residual. Adding ratio features to the forest detected 67 of 98 anomalies while producing 128 false alerts. Its test AP improvement over the C forest was +0.020, with a paired 95% interval of −0.036 to +0.102. The test split was reused for comparison, so this estimate is exploratory.
