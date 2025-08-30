# Loan Default Prediction using Machine Learning

This repository contains my assignment project where I developed a **machine learning pipeline** to predict loan approval status based on customer financial and demographic details.  

The project walks through the **complete ML lifecycle**:  
- Importing and cleaning data  
- Exploratory Data Analysis (EDA)  
- Preprocessing and feature engineering  
- Training multiple models (Random Forest, SVM)  
- Model evaluation using classification metrics  

---

## 📖 Problem Statement
Financial institutions often struggle to assess whether a loan applicant is likely to **repay** or **default**. Manual assessments can be biased and inconsistent.  

By applying machine learning, we can build a **predictive model** that uses past loan applicant data to make **data-driven, consistent, and accurate predictions** of loan approval.

---

## 📊 Dataset
The dataset (`loan_data.csv`) contains applicant information with attributes such as:  
- **Applicant details** → Gender, Marital Status, Dependents, Education, Employment status  
- **Financial attributes** → Applicant Income, Co-applicant Income, Loan Amount, Loan Term, Credit History  
- **Loan outcome** → Loan Status (Approved / Not Approved)  

The target variable is **Loan Status** (binary classification).  

---

## ⚙️ Workflow

### 1. Importing Libraries
Libraries used include:
- `pandas`, `numpy` → data handling  
- `matplotlib`, `seaborn` → visualization  
- `scikit-learn` → preprocessing, model training & evaluation  

---

### 2. Data Preprocessing
- Loaded the dataset into a Pandas DataFrame.  
- Handled **missing values** using appropriate imputation.  
- Converted categorical variables (Gender, Education, Self_Employed, Loan_Status) into numerical values using **Label Encoding**.  
- Standardized numerical features (ApplicantIncome, LoanAmount, etc.) using **StandardScaler** to ensure consistent scaling.  

---

### 3. Exploratory Data Analysis (EDA)
- **Correlation Matrix** → Visualized relationships between features.  
- **Distribution Analysis** → Plotted histograms for income, loan amount, and credit history.  
- **Target Relationship** → Compared loan approval rates across categorical groups (e.g., education, marital status).  
- Key insight: **Credit history** strongly influences loan approval.  

---

### 4. Model Building
Two main models were trained and tested:  
1. **Random Forest Classifier**  
   - Captures feature importance and non-linear relationships.  
   - Tuned using **GridSearchCV** for hyperparameters like `n_estimators`, `max_depth`.  

2. **Support Vector Classifier (SVM)**  
   - Tested with linear and RBF kernels.  
   - Tuned parameters like `C` and `gamma` using GridSearchCV.  

Data was split into **train/test sets** to validate model performance.  

---

### 5. Model Evaluation
Evaluated using multiple metrics:  
- **Accuracy Score** → overall correctness  
- **Confusion Matrix** → counts of true positives/negatives, false positives/negatives  
- **Precision & Recall** → useful for imbalanced classes  
- **F1 Score** → balance between precision and recall  
- **ROC Curve & AUC** → classification threshold performance  

---

## 📈 Results

- **Random Forest Classifier** achieved the best overall performance.  
- Example results (numbers depend on dataset split):  
  - Accuracy: ~82–85%  
  - Precision: ~0.82  
  - Recall: ~0.84  
  - F1-score: ~0.83  
  - ROC-AUC: ~0.87  

- **SVM** also performed well but required careful tuning of hyperparameters.  

---

## 🔑 Key Insights
- **Credit History** was the most important predictor of loan approval.  
- Applicant Income and Loan Amount also contributed significantly.  
- Random Forest provided **better interpretability and robustness** compared to SVM.  
- Visualizations revealed clear differences in loan approval rates based on financial and demographic attributes.  

---

## 🌍 Impact and Applications
- Helps financial institutions make **automated, consistent loan decisions**.  
- Reduces risk of defaults by identifying high-risk applicants.  
- Improves customer experience with **faster loan processing**.  
- Demonstrates practical application of **machine learning in finance**.  

---

## 🛠️ Tools & Libraries
- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## 📂 Project Structure
