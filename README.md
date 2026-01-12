# Public Health Risk Prediction System

## Overview
This project is an end-to-end machine learning system that classifies regions
into **High Risk** and **Low Risk** categories using government healthcare data.

The objective is to assist policymakers and public health authorities in
identifying regions that require priority healthcare intervention.

---

## Problem Statement
Government healthcare data is often underutilized in proactive decision-making.
This project uses machine learning to analyze public health indicators and
predict regional healthcare risk levels to support data-driven policy planning.

---

## Dataset
The project uses publicly available government healthcare datasets
(World Health Organization / World Bank), which include indicators such as:
- Life expectancy
- Infant mortality rate
- Health expenditure
- Population statistics

---

## Machine Learning Approach
- Data preprocessing and cleaning
- Feature selection and scaling
- Supervised learning models
- Model evaluation using classification metrics

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

## Project Structure
public-health-risk-ml/
├── app/ # Streamlit application
├── data/ # Dataset files
├── model/ # Training scripts
├── notebooks/ # EDA notebooks
├── requirements.txt
└── README.md

---

## Project Status
**Phase 1 – MVP under development**

---

## Future Improvements
- Incorporate additional healthcare indicators
- Improve feature engineering
- Add explainability (SHAP)
- Deploy the system on cloud infrastructure
