# ESI Prediction Health Care System

## Overview

This project develops an AI-driven triage prediction system to enhance decision-making in Emergency Departments (EDs). It employs supervised machine learning to assign Emergency Severity Index (ESI) levels (1 to 5, from most to least urgent) based on patient data, including demographics, vital signs, medical history, and chief complaints. A web-based application built with **Flask** enables real-time clinical use, tailored to the Romanian medical context.

---

## Dataset

### Source and Description

The dataset is a publicly available collection from three hospitals, comprising over **500,000 patient records**. Each record includes:

* **Demographics**: Age, gender
* **Vital Signs**: Heart rate, blood pressure, respiratory rate, temperature
* **Medical History**: Past diagnoses, medications
* **Chief Complaints**: Reason for ED visit

The **target** is the ESI level (1 = most urgent, 5 = least urgent), distributed as:

| ESI Level | Records |
| --------- | ------- |
| Level 1   | 4,265   |
| Level 2   | 130,127 |
| Level 3   | 182,245 |
| Level 4   | 95,255  |
| Level 5   | 22,107  |

This real-world class imbalance required special handling during preprocessing.

---

## Data Cleaning and Preprocessing

| Step                          | Description                               | Tool/Library                  |
| ----------------------------- | ----------------------------------------- | ----------------------------- |
| Dropping Missing Labels       | Removed records without ESI labels        | pandas                        |
| Imputing Numerical Values     | Filled missing values with median         | pandas                        |
| Encoding Categorical Features | One-hot encoded categorical columns       | scikit-learn (OneHotEncoder)  |
| Scaling Numerical Data        | Standardized features (mean=0, std=1)     | scikit-learn (StandardScaler) |
| Addressing Class Imbalance    | Used SMOTE to oversample minority classes | imblearn (SMOTE)              |

---

## Clustering Analysis

Used **K-Means (k=3)** to group patients by age, ESI level, and admission frequency. Visualized using **PCA**:

* **Cluster 0 - Younger, Less Severe**:

  * Avg Age: 34.37
  * Mean ESI: 3.19
  * Avg Admissions: 0.30
  * Description: Non-critical first-time ED visitors

* **Cluster 1 - Middle-Aged, Intermediate Severity**:

  * Avg Age: \~52
  * Mean ESI: Intermediate
  * Description: Chronic/intermediate patients needing timely care

* **Cluster 2 - Elderly, High Severity**:

  * Avg Age: Elderly
  * Mean ESI: High (1-2)
  * Description: Urgent and frequent ED visitors

---

## Models

Several supervised ML models were tested. The **Sequential Neural Network (SNN)** yielded the best performance.

| Model              | Description                   | Notes                            |
| ------------------ | ----------------------------- | -------------------------------- |
| KNN                | Distance-based classification | Sensitive to feature scaling     |
| SVM                | Margin-based classification   | Great for high-dimensional data  |
| Random Forest      | Ensemble of decision trees    | Robust to overfitting            |
| AdaBoost           | Boosting algorithm            | Enhances weak learners           |
| XGBoost            | Gradient boosting             | High performance on tabular data |
| Ordinal Regression | Predicts ordered categories   | Captures ESI order relationships |
| SNN                | Deep learning                 | Highest accuracy in benchmarks   |

---

## Results

Models were evaluated on accuracy, precision, recall, F1-score, and ROC-AUC. The SNN model performed the best with high accuracy across all metrics.

---

## Contributors

**Ahmed Nour Allah Mejri**

---

## License

License not specified.

---

## Citations

* GitHub Repository
* Project Documentation: *Medical Emergency Department Triage Data Processing Using Supervised Machine Learning Models* by Ahmed Nour Allah Mejri
