# supervised-creditworthiness
using a logistic regression model to better understand credit

---

## Overview of the Analysis

* The purpose of the analysis was to see if we could use supervised learning to find patterns and build a better predictor for classifcation of loans, healthy and non-healthy. 
* The data consistsed of 77k loans and a schema of loan_size, interest_rate, borrower_income, debt_to_income ratio, number_of_accounts, derotatory_marks, total_debt, loan_status
* The main predictor is if a loan is healthy or not 
* Prepare and clean the data. Split and test the data. Fit the model, run it, then evaulate the findings 
* It was a logistic regression model that leveraged an accuracy_score from sklearn

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Orginal Data Model:
 * Accuracy score: 0.952048
 * Precision: Class 0: 1.00 Class 1: 0.85
 * Recall: Class 0: 0.99 Class 1: 0.91

* Machine Learning Resampled Data Model:
 * Accuracy score: 0.993678
 * Precision: Class 0: 1.00 Class 1: 0.84
 * Recall: Class 0: 0.99 Class 1: 0.99

## Summary

Based on the findings, I would select the machine learning resampled data model for the main reason that the accuracy score was higher and the recall was imporved for class 1. They're both strong models and more should be explored.

---
## Tools 

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

---
## Please find the code [credit_risk](https://github.com/Brock-Denton/supervised-creditworthiness/blob/main/credit_risk_resampling-checkpoint.ipynb)
---
## Contributor
### Brock Denton, Brockchecksmail@gmail.com 
---
### License 
MIT 
