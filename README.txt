# Titanic Survival Prediction

This project uses **Logistic Regression** to predict passenger survival on the Titanic using the [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic).

## Overview
- Train a model to predict `Survived`.
- Handle missing data (`Age`, `Embarked`, `Cabin`).
- Encode categorical variables (`Sex`, `Embarked`).
- Evaluate with accuracy, confusion matrix, and classification report.
- Generate Kaggle submission file.

## Data Preprocessing
- Fill `Age` with median, `Embarked` with mode.
- Drop `Cabin` and irrelevant columns (`PassengerId`, `Name`, `Ticket`).
- Convert `Sex` to numeric and one-hot encode `Embarked`.

## Model
- Logistic Regression trained on 80% of data, validated on 20%.
- Evaluate performance with accuracy and confusion matrix.

## Kaggle Submission
- Apply same preprocessing to test set.
- Predict `Survived` and save as `titanic_submission.csv`.

## Requirements
- Python 3.x
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
