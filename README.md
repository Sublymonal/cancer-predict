# cancer-predict
## Make sure you are able to run the following libraries
- seaborn
- matplotlib
- pandas
- sklearn
- lime
- numpy
- plotly




# 🩺 Breast Cancer Prediction Web App

This project is a full-stack machine learning web application that predicts whether a breast tumor is benign or malignant based on diagnostic features. Built using Python, Flask, and deployed on Google Cloud App Engine, it integrates machine learning, interactive data visualizations, and model interpretability tools to support early cancer detection.

- Edit: App would not deploy properly, Streamlit workaround to run locally
## Utility to install
<p>- Streamlit</p>

The purpose of this repo is to test breast cancer tumors for malignancy with machine learning prediction.<p>
Website: https://advance-archery-458203-t3.uc.r.appspot.com/

<p>To try prediction app, use command streamlit run app2.py</p><p></p>
---

## 📊 Project Overview

The application uses a **Random Forest Classifier** trained on a breast cancer dataset from Kaggle. It supports:

- Interactive prediction using user-controlled sliders.
- Visualization of classification outcomes (TP, TN, FP, FN).
- LIME-based model interpretation for transparency.
- Real-time web deployment via Google Cloud.

---

## 🧠 Machine Learning Pipeline

- **Data Preprocessing**:
  - Cleaned dataset with no missing values.
  - Applied `StandardScaler` for feature normalization.

- **Model**:
  - `RandomForestClassifier` trained to distinguish between malignant (0) and benign (1) cases.
  - Focused on minimizing false negatives, critical in cancer detection.

- **Interpretability**:
  - Used **LIME** to understand why the model made incorrect predictions.
  - Special attention given to cases where malignant tumors were predicted as benign.

---

## 🖼️ Pages and Features

- **Home**: Project summary, random forest tree image.
- **Objectives**: Project goals and success criteria.
- **Methods**: Technical explanation of the tools and approaches used.
- **Findings**: Analysis of accuracy, false negatives, and model transparency.
- **Predict**: Interactive prediction tool with sliders and confidence output.

---

## 🌐 Deployment

Deployed using **Google Cloud App Engine**. The following resources were used:

- `Flask` for backend routing
- `Plotly.js` for interactive charts
- `gunicorn` for WSGI serving
- `joblib` for loading trained ML model

---

## 🛠️ Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Sublymonal/cancer-predict.git
   cd cancer-prdict
