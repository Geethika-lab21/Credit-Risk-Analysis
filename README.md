# Credit Risk Analysis

## 📌 Overview

This project predicts loan default risk using machine learning and data analysis techniques. It combines SQL, EDA, and ML to generate actionable business insights.

---

## 🎯 Problem Statement

Financial institutions face significant losses due to loan defaults. The goal is to identify high-risk borrowers before approving loans.

---

## 📊 Dataset

* Lending Club dataset (~1.19GB)
* Used stratified sampling to create a balanced dataset of 200,000 records

---

## 🛠️ Tech Stack

* Python (Pandas, NumPy, Scikit-learn)
* SQL (PostgreSQL)
* Visualization (Seaborn, Matplotlib)

---

## ⚙️ Workflow

1. Data Sampling (balanced dataset)
2. Data Cleaning & EDA
3. SQL-based analysis
4. Machine Learning (Logistic Regression, Random Forest)
5. Model evaluation

---

## 📈 Results

* Random Forest achieved best performance
* Key predictors: DTI, interest rate, credit behavior

---

## 💡 Business Insights

* High DTI significantly increases default risk
* Higher interest rates correlate with risky borrowers
* Credit utilization plays a major role in loan repayment

---

## 📂 Project Structure

* notebooks/ → analysis and modeling
* sql/ → queries and schema
* models/ → trained ML model

---

## 🚀 Future Improvements

* Deploy model using Streamlit
* Add explainable AI (SHAP)
* Use real-time API integration


