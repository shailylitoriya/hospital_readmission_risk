# Hospital Readmission Prediction

This project predicts the likelihood of patient readmission within 30 days using the `diabetic_data.csv` dataset. It involves data cleaning, EDA, outlier removal, feature engineering, handling imbalanced data using SMOTE, and training with XGBoost and hyperparameter tuning.

## Structure

- `diabetics.csv`: Raw dataset
- `Hospital Readmission Project.ipynb`: Jupyter notebook with EDA and model training
- `outputs`: Saved plots from EDA and model evaluation

## Key Steps

- Outlier removal
- Encoding categorical features
- SMOTE for class imbalance
- XGBoost with GridSearchCV tuning
- Threshold tuning for better recall

## Project Workflow

### 1. **Data Preprocessing**
- Removed outliers from key numerical features such as:
  - `num_lab_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`
- Handled missing values and replaced garbage values
- Encoded categorical features using Label Encoding
- Created engineered features like:
  - `medication_bins`, `stay_length`, and `is_elderly`

### 2. **Exploratory Data Analysis (EDA)**
- Count plots and distribution plots for:
  - Readmission status
  - Medication levels
  - Hospital stay duration
  - Age groups
- Correlation heatmap to identify important features

### 3. **Feature Engineering**
- Binned medication and stay duration
- Created additional informative features based on domain logic

### 4. **Model Building**
- Used `XGBoostClassifier` with class imbalance handling (`scale_pos_weight`)
- Applied SMOTE for oversampling the minority class
- Performed hyperparameter tuning using `GridSearchCV` with StratifiedKFold

### 5. **Model Evaluation**
- Evaluated model using:
  - Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Tuned classification threshold (e.g., 0.6) for improved recall
- Plotted Precision-Recall vs. Threshold curve

---

## Results

- **Final Model:** XGBoost (with SMOTE + Hyperparameter Tuning)
- **Threshold Applied:** 0.6

## Final Model Performance (Threshold = 0.6)

- Accuracy: 81%
- Precision (class 1): 0.22
- Recall (class 1): 0.31
- F1-Score (class 1): 0.26
- **Confusion Matrix:**
  [[7339 1127]
 [ 694  314]]

---

## How to Use

1. Clone the repo:
   ```bash
   git clone https://github.com/shailylitoriya/hospital_readmission_risk.git
   cd hospital_readmission_risk
   ```
   
2. Install with:

```bash
pip install -r requirements.txt
```

## License
This project is for educational purposes only.

## Author
Shaily Litoriya
[Github](https://github.com/shailylitoriya) [LinkedIn](www.linkedin.com/in/shailylitoriya)
