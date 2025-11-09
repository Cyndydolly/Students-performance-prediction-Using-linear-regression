# 🎓 Student Performance Prediction

This project uses **Machine Learning** to predict student academic performance based on multiple factors such as gender, parental education, and test preparation.  
It demonstrates data cleaning, visualization, feature engineering, and predictive modeling — a full data science workflow.

---

## 📊 Project Overview

The goal of this project is to analyze and predict how various socio-economic and personal factors affect a student's performance in exams.  
We use both **Regression** (predicting scores) and **Classification** (Pass/Fail) techniques.

---

## 🧠 Objectives

- Explore relationships between different student attributes and exam performance  
- Build a predictive model for student scores  
- Evaluate model accuracy using regression metrics (R², MAE, MSE)  
- (Optional) Classify students as *Pass* or *Fail* using classification models  

---

## ⚙️ Technologies Used

| Category | Tools |
|-----------|-------|
| Programming | Python |
| Libraries | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn |
| Environment | Jupyter Notebook / VS Code |
| Version Control | Git & GitHub |

---

## 🚀 Project Workflow

1. **Data Loading**  
   - Imported dataset from [Kaggle: Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

2. **Exploratory Data Analysis (EDA)**  
   - Checked for missing values, outliers, and distributions  
   - Visualized relationships using Seaborn and Matplotlib  

3. **Data Preprocessing**  
   - Encoded categorical features  
   - Created a new column `average` (mean of math, reading, writing scores)  

4. **Model Building**  
   - Regression: Linear Regression to predict average score  
   - Classification: Random Forest Classifier for Pass/Fail prediction  

5. **Model Evaluation**  
   - Regression: MAE, MSE, R² Score  
   - Classification: Accuracy, Precision, Recall, Confusion Matrix  

---

## 📈 Results Summary

| Model | Type | Metric | Score |
|--------|------|--------|-------|
| Linear Regression | Regression | R² Score | 0.88 |
| Random Forest | Classification | Accuracy | 0.90 |

✅ The regression model explained **88% of performance variation**, and the classification model correctly predicted **90%** of student outcomes.

---

## 📁 Folder Structure

