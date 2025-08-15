# 🩺 Diabetes Classification Pipeline

A complete end-to-end **Machine Learning pipeline** for classifying patients as **diabetic** or **non-diabetic** using the PIMA Indians Diabetes Dataset.  
This project demonstrates **data preprocessing, outlier handling, feature scaling, model training, hyperparameter tuning, cross-validation, and performance evaluation** across multiple classifiers.

---

## 📌 Project Overview
The goal of this project is to build and evaluate multiple supervised learning models to predict diabetes status from patient health measurements.  
The dataset consists of 768 samples and 9 columns, including numerical features such as glucose concentration, BMI, and insulin levels.

---

## ⚙️ Features
- **Data Preprocessing**
  - Outlier detection & handling (IQR method)
  - Missing/zero value handling for physiological features
  - Feature scaling using `StandardScaler`
- **Data Splitting & Cross-Validation**
  - Train/Validation/Test split
  - Stratified K-Fold cross-validation
- **Hyperparameter Tuning**
  - Grid Search on key parameters for each model
- **Model Training**
  - SVM (Linear, Polynomial, RBF, Sigmoid)
  - Decision Tree
  - Random Forest
  - XGBoost
  - k-NN
  - Logistic Regression
- **Evaluation Metrics**
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrices (heatmaps)
  - Summary comparison table
- **Visualization**
  - Model comparison bar charts
  - Confusion matrix heatmaps for each classifier

---

## 📊 Results Summary
| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| SVM                 | 0.712    | 0.700     | 0.712  | 0.695    |
| Decision Tree       | 0.699    | 0.684     | 0.699  | 0.678    |
| Random Forest       | 0.699    | 0.689     | 0.699  | 0.691    |
| XGBoost             | 0.699    | 0.689     | 0.699  | 0.691    |
| **k-NN**            | **0.725**| **0.721** | **0.725** | **0.723** |
| Logistic Regression | **0.725**| 0.715     | **0.725** | 0.709    |

---

## 📁 Repository Structure
```

├── data/                     # Dataset (if included) or link in README
├── notebooks/                # Jupyter notebooks / .py scripts
└── README.md                 # Project documentation

````

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/diabetes-classification.git
cd diabetes-classification
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the notebook or script

```bash
jupyter notebook
```

or

```bash
python main.py
```

---

## 📦 Dependencies

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* jupyter (if running notebook)

Install them all with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

---

## 📌 Conclusion

* **Best overall performers:** k-NN and Logistic Regression (\~72.5% accuracy).
* **Improvement opportunities:** Address class imbalance (e.g., SMOTE, class weighting).
* Next steps could involve feature engineering, ensemble methods, or deep learning approaches.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Ali Badran**
B.Sc. Systems & Biomedical Engineering | AI & Deep Learning Enthusiast
[LinkedIn](https://www.linkedin.com/in/ali-badran) | [GitHub](https://github.com/AliBadran716)

