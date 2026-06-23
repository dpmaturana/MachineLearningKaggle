# Machine Learning Kaggle Labs

A collection of Jupyter notebooks built as labs/homeworks for Machine Learning coursework, each tackling a different Kaggle competition with a different modeling challenge (imbalanced classification, multi-class stacking, tabular feature engineering, SVM tuning).

---

## Santander Customer Transaction Prediction
**`Santander_.ipynb`**

🔗 [Kaggle competition](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)

Binary classification task: predict which customers will make a specific transaction, using 200 anonymized numeric features. Evaluated on AUC-ROC.

**Approach:** LightGBM with 5-fold stratified cross-validation, tuned via Optuna. Detects which test rows are "real" vs. synthetic (a known quirk of this dataset) to build undistorted count-based features, combined with row-wise statistics (mean, std, min/max, unique count). Final predictions are blended across multiple random seeds, then refined with conservative pseudo-labeling (only confident predictions above/below a threshold are added back as training data).

---

## Porto Seguro's Safe Driver Prediction
**`Porto_seguro.ipynb`**

🔗 [Kaggle competition](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)

Imbalanced binary classification: predict the probability a driver files an insurance claim next year. Evaluated on the Normalized Gini Coefficient.

**Approach:** Feature engineering tailored to the dataset's quirks — dropping low-signal `calc_` features, encoding `-1` (missing) counts as a feature, combining categorical indicators into interaction strings, and applying both count encoding and one-hot encoding to categorical variables. Final model is LightGBM with class-weighting for imbalance, tuned with Bayesian Optimization over leaf count, regularization, and sampling fractions.

---

## Costa Rican Household Poverty Level Prediction
**`Costa_Rica_Poverty.ipynb`**

🔗 [Kaggle competition](https://www.kaggle.com/c/costa-rican-household-poverty-prediction)

Multi-class classification (4 poverty levels) predicting household need for social welfare assistance. Evaluated on Macro F1-score.

**Approach:** Fixes label inconsistencies within households (assigns the head-of-household's target to all members), engineers composite poverty indices (e.g. combined wall/roof/floor quality scores), and addresses class imbalance with SMOTE applied strictly inside each cross-validation fold to avoid data leakage. Core model is a stacked ensemble (LightGBM + XGBoost + CatBoost as base learners, feeding out-of-fold predictions into a Logistic Regression meta-learner).

---

## Spam/Ham Email Classifier
**`Spam_Ham_Detector.ipynb`**

🔗 [Kaggle InClass competition — UC Berkeley CS189 HW1 (SPAM), Spring 2024](https://www.kaggle.com/competitions/cs189-hw1-spam-spring-2024)

Binary classification: spam vs. ham on a pre-featurized dataset (32 numeric features per email), evaluated on accuracy. Part of UC Berkeley's CS189 (Intro to Machine Learning) coursework.

**Approach:** Features are scaled with `StandardScaler`, then an SVM (`sklearn.svm.SVC`) is trained and evaluated via held-out validation and 5-fold cross-validation. Hyperparameters (kernel, `C`, class weighting) are tuned and compared across two search strategies — `RandomizedSearchCV` and Optuna — keeping whichever yields the higher CV accuracy. The final model is refit on the full training set before generating Kaggle-formatted test predictions.

---

## Stack

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · CatBoost · Optuna · imbalanced-learn (SMOTE) · matplotlib · seaborn
