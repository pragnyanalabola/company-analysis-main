import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

# Load dataset (Example: Financial Data of a Company)
df = pd.read_csv('company_data.csv')  # Replace with actual dataset

# Streamlit Frontend
st.title("Company Analysis Dashboard")

# Data Overview
st.subheader("Data Preview")
st.write(df.head())

st.subheader("Dataset Information")
st.text(str(df.info()))

st.subheader("Statistical Summary")
st.write(df.describe())

# Handling missing values
df.fillna(df.mean(), inplace=True)

# Exploratory Data Analysis (EDA)
st.subheader("Pairplot Analysis")
st.pyplot(sns.pairplot(df))

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
st.pyplot(fig)

# Revenue Trend Analysis
st.subheader("Company Revenue Over Time")
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=df, x='Year', y='Revenue', marker='o', ax=ax)
st.pyplot(fig)

# Machine Learning: Predicting Future Revenue
X = df[['Marketing_Spend', 'R&D_Spend', 'Operational_Cost']]  # Independent Variables
y = df['Revenue']  # Dependent Variable

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

st.subheader("Model Performance Metrics")
st.write(f'MAE: {mae}')
st.write(f'MSE: {mse}')
st.write(f'RMSE: {rmse}')

# Future Prediction Example
st.subheader("Future Revenue Prediction")
future_data = np.array([[50000, 20000, 30000]])  # Example input data
predicted_revenue = model.predict(future_data)
st.write(f'Predicted Future Revenue: {predicted_revenue[0]}')
