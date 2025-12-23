# Insurance Charges Prediction using Polynomial Regression

## 📌 Project Overview
This project analyzes the **Insurance dataset** to predict medical insurance charges based on customer attributes such as age, BMI, smoking status, and region.

Since the relationship between features (especially BMI) and insurance charges is **non-linear**, both **Linear Regression** and **Polynomial Regression** models are implemented and compared.

---

## 📊 Dataset Description
The dataset contains the following features:

| Feature | Description |
|-------|------------|
| age | Age of the insured person |
| sex | Gender (male/female) |
| bmi | Body Mass Index |
| children | Number of children covered |
| smoker | Smoking status (yes/no) |
| region | Residential region |
| charges | Medical insurance cost (target variable) |

---

## 🧪 Steps Performed

### 1️⃣ Data Exploration (EDA)
- Displayed dataset structure and summary statistics
- Checked for missing values
- Visualized **BMI vs Charges** with smoker status

### 2️⃣ Data Preprocessing
- Converted categorical variables (`sex`, `smoker`, `region`) into numeric values using **Label Encoding**
- Ensured dataset is fully numerical for regression models

### 3️⃣ Model Training
- Split data into **80% training** and **20% testing**
- Trained:
  - **Linear Regression (Degree 1)**
  - **Polynomial Regression (Degree 2)**

### 4️⃣ Model Evaluation
Used the following metrics:
- **RMSE (Root Mean Squared Error)**
- **R² Score**

A comparison table and bar chart were generated to evaluate performance.

### 5️⃣ Overfitting Analysis
- Trained Polynomial Regression models with degrees **1 to 4**
- Plotted **Model Complexity vs Error**
- Identified **Degree 2** as the optimal balance between bias and variance

---

## 📈 Visualizations Included
- BMI vs Charges scatter plot (colored by smoker status)
- RMSE comparison bar chart
- Actual vs Predicted charges plot
- Polynomial degree vs RMSE (Overfitting analysis)

---

## 🏆 Results Summary
- Polynomial Regression (Degree 2) outperforms Linear Regression
- Higher degrees lead to overfitting
- Smoking status has a strong impact on insurance charges

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 🚀 How to Run
1. Clone the repository
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
