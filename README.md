# Santander Customer Transaction Prediction

Predicting which customers will make a specific transaction, from 200 anonymous numeric features. Coursework for Advanced Machine Learning at IE School of Science and Technology (Master in Computer Science and Business Technology, 2026), built on the Kaggle competition of the same name.

## Data

`train.csv` and `test.csv` come from the [Kaggle competition](https://www.kaggle.com/competitions/santander-customer-transaction-prediction) and are not included. Place them in `data/`.

- Train: 200,000 rows, 200 features, 10.05% positive class
- Test: 200,000 rows, of which 100,000 are real and 100,000 are synthetic (see notebook 01b)

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `01_eda.ipynb` | Per-class feature distributions and train/test comparison |
| `01b_synthetic_reconstruction.ipynb` | Separates real test rows from synthetic ones |
| `02_naive_bayes.ipynb`, `03_nb_experiments.ipynb`, `04_naive_bayes_final.ipynb` | Naive Bayes baseline and variants |
| `05_lightgbm.ipynb`, `06_lightgbm_experiments.ipynb` | LightGBM baseline, Optuna tuning, multi-seed blend |

## Results (out-of-fold AUC, five folds)

| Model | AUC |
| --- | --- |
| Naive Bayes | 0.890 |
| LightGBM, default parameters | 0.905 |
| LightGBM, Optuna-tuned | 0.910 |
| LightGBM, three-seed blend | 0.911 |

## Stack

Python, LightGBM, Optuna, scikit-learn, pandas, NumPy, matplotlib.
