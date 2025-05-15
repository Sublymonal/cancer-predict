{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "525808ac-40f7-4028-baec-bc6fb6275bdc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import joblib\n",
    "import pandas as pd\n",
    "\n",
    "# Load the trained model and scaler\n",
    "model = joblib.load(\"optimized_model.pkl\")  # Make sure this file exists\n",
    "scaler = joblib.load(\"scaler.pkl\")  # Make sure this file exists\n",
    "\n",
    "# Define feature names based on dataset\n",
    "feature_names = [\n",
    "    \"mean_radius\", \"mean_texture\", \"mean_perimeter\",\n",
    "    \"mean_area\", \"mean_smoothness\"\n",
    "]\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"Breast Cancer Prediction App 🎛️🔮\")\n",
    "st.write(\"Adjust the sliders below to update the prediction in real-time.\")\n",
    "\n",
    "# Create sliders for each feature\n",
    "user_input = {}\n",
    "for feature in feature_names:\n",
    "    user_input[feature] = st.slider(\n",
    "        feature, float(0), float(50), float(25), step=0.1\n",
    "    )\n",
    "\n",
    "# Convert input into a DataFrame\n",
    "input_data = pd.DataFrame([user_input])\n",
    "\n",
    "# Scale the input using the same scaler as training\n",
    "input_scaled = scaler.transform(input_data)\n",
    "\n",
    "# Predict using the trained model\n",
    "prediction = model.predict(input_scaled)[0]\n",
    "prediction_proba = model.predict_proba(input_scaled)[0]\n",
    "\n",
    "# Display prediction result\n",
    "st.subheader(\"Prediction Result:\")\n",
    "if prediction == 1:\n",
    "    st.markdown(\"### 🚨 Malignant (Cancerous)\")\n",
    "    st.write(f\"Confidence: {prediction_proba[1]*100:.2f}%\")\n",
    "else:\n",
    "    st.markdown(\"### ✅ Benign (Non-Cancerous)\")\n",
    "    st.write(f\"Confidence: {prediction_proba[0]*100:.2f}%\")\n",
    "\n",
    "# Run Streamlit with: streamlit run app.py\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c4693242-a458-42c0-82ba-5b2c45abb489",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
