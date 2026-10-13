# Customer Churn Prediction and Probability Calibration

A reproducible Google Colab machine learning experiment comparing Logistic Regression and Random Forest for predicting customer churn, with emphasis on **average precision (PR AUC)**, **recall**, and **probability calibration**. A prevalence-based Dummy Classifier provides a baseline.

## Dataset

- **Source:** [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **CSV:** [Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Size:** 7,043 customer records, 21 original columns
- **Target:** `Churn` (`Yes` = 1; `No` = 0)
- **Dataset access date:** *Add the actual access date before publication.*
- **Usage terms:** *Verify the dataset-specific reuse terms before publication; see [`data/README.md`](data/README.md).*

The notebook downloads the CSV at runtime; no local dataset file is required.

## Methodology

1. Inspect missing values, identifiers and churn distribution. Convert `TotalCharges` to numeric.
2. Exclude `customerID` and split data into stratified **60% training / 20% validation / 20% held-out test**, using `random_state=42`.
3. Fit preprocessing within scikit-learn pipelines: median imputation and standardization for numeric features; categorical imputation and one-hot encoding for categorical features.
4. Compare a Dummy Classifier, Logistic Regression and Random Forest.
5. Tune model hyperparameters on validation average precision.
6. Assess precision–recall trade-offs and select thresholds on validation data with a precision floor of 0.40.
7. Compare original and five-fold sigmoid-calibrated probabilities using validation Brier score, then select final thresholds on validation data.
8. Evaluate finalized models once on the held-out test split and export results and figures.

**Interpretation caution:** The validation set supports several choices, so validation-based selection may be optimistic. Test scores should not be used for additional tuning.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Upload notebook** and upload `Customer_Churn_Project.ipynb`.
3. Select a Python CPU runtime and choose **Runtime → Run all**. Internet access is required to retrieve the IBM CSV.
4. Download the generated `figures/` and `results/` folders from the Colab file browser for inclusion in the GitHub repository.

## Run locally

Python 3.10+ is recommended. In a terminal:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install jupyterlab
jupyter lab Customer_Churn_Project.ipynb
```

Run all notebook cells in order. A notebook environment must support `display()` (Colab/Jupyter). The exact library versions and runtime are written to `results/experiment_info.json` when the notebook completes.

## Outputs

- [`results/`](results/): validation metrics, tuning comparisons, final three-model test comparison and experiment metadata.
- [`figures/`](figures/): churn distribution, calibration diagrams, threshold analysis, precision–recall curves and confusion matrices.

The repository package initially contains the **code and output directories**, not executed output files. Run the notebook and add the resulting CSV/JSON/PNG files before final submission.

## Model evaluation

The final notebook calculates **average precision**, **recall**, **precision**, **Brier score**, and confusion matrices on the held-out test data. Average precision summarizes the precision–recall curve; Brier score assesses probability accuracy (lower is better). Recall and precision depend on the chosen classification threshold.

## Limitations

The analysis uses one public dataset and one train/validation/test split. Threshold selection targets a precision constraint rather than measured retention costs. Some false-positive alerts may therefore be expensive in practice. Results should not be generalized to other populations without additional validation.

## Project files

```text
Customer_Churn_Project.ipynb
README.md
requirements.txt
.gitignore
data/README.md
figures/README.md
results/README.md
```
