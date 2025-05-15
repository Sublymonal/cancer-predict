import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and scaler
model = joblib.load("optimized_model.pkl")  # Ensure this file exists
scaler = joblib.load("scaler.pkl")  # Ensure this file exists

# Load the original dataset (for visualization purposes)
df = pd.read_csv("data.csv")  # Ensure this file exists

# Separate features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Scale the dataset
X_scaled = scaler.transform(X)

# Append diagnosis back for visualization
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled['diagnosis'] = y

# Define feature names
feature_names = [
    "mean_radius", "mean_texture", "mean_perimeter",
    "mean_area", "mean_smoothness"
]

# Streamlit UI
st.title("Breast Cancer Prediction App")
st.write("Adjust the sliders below to update the prediction in real-time.")

# Create sliders for each feature
user_input = {}
for feature in feature_names:
    user_input[feature] = st.slider(
        feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()), step=0.1
    )

# Convert input into a DataFrame
input_data = pd.DataFrame([user_input])

# Scale the input using the same scaler as training
input_scaled = scaler.transform(input_data)

# Predict using the trained model
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0]

# Display prediction result
st.subheader("Prediction Result:")
if prediction == 1:
    st.markdown("### Benign (Non-Cancerous)")
    st.write(f"Confidence: {prediction_proba[1]*100:.2f}%")
else:
    st.markdown("### Malignant (Cancerous)")
    st.write(f"Confidence: {prediction_proba[0]*100:.2f}%")

# Visualization: Real-time Chart
st.subheader("📊 Data Distribution & Your Input")

# Select two key features for visualization
feature_x = "mean_radius"
feature_y = "mean_perimeter"

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df_scaled[feature_x],
    y=df_scaled[feature_y],
    hue=df_scaled["diagnosis"],
    palette={0: "red", 1: "green"},
    alpha=0.6
)

# Plot user's input as a large blue dot
plt.scatter(input_scaled[0][0], input_scaled[0][2], color="blue", s=150, label="Your Input", edgecolors="black")

plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title("Malignant vs. Benign Distribution")
#plt.legend("Benign (Green)", "Malignant (Red)")
st.pyplot(plt)

# Select two key features for visualization
feature_x = "mean_area"
feature_y = "mean_perimeter"

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df_scaled[feature_x],
    y=df_scaled[feature_y],
    hue=df_scaled["diagnosis"],
    palette={0: "red", 1: "green"},
    alpha=0.6
)

# Plot user's input as a large blue dot
plt.scatter(input_scaled[0][0], input_scaled[0][3], color="blue", s=150, label="Your Input", edgecolors="black")

plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title("Malignant vs. Benign Distribution")
#plt.legend(["Benign (Green)", "Malignant (Red)", "Your Input (Blue)"])
st.pyplot(plt)

# Select two key features for visualization
feature_x = "mean_area"
feature_y = "mean_radius"

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df_scaled[feature_x],
    y=df_scaled[feature_y],
    hue=df_scaled["diagnosis"],
    palette={0: "red", 1: "green"},
    alpha=0.6
)

# Plot user's input as a large blue dot
plt.scatter(input_scaled[0][3], input_scaled[0][0], color="blue", s=150, label="Your Input", edgecolors="black")

plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title("Malignant vs. Benign Distribution")
#plt.legend(["Benign (Green)", "Malignant (Red)", "Your Input (Blue)"])
st.pyplot(plt)


