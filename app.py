import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# Title
st.title("🏠 House Price Prediction App")
st.write("Simple Linear Regression using House Area")

# Load dataset
data = pd.read_csv("simple_linear_house_dataset.csv")

# Show dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Features and target
X = data[['Area_sqft']]
y = data['Price_INR']

# Train model
model = LinearRegression()
model.fit(X, y)

# Sidebar input
st.sidebar.header("Enter House Area")

area = st.sidebar.number_input(
    "Area in Square Feet",
    min_value=500,
    max_value=5000,
    value=1000
)

# Prediction
if st.sidebar.button("Predict Price"):
    
    prediction = model.predict([[area]])
    
    st.subheader("Predicted House Price")
    st.success(f"Estimated Price: ₹ {int(prediction[0])}")

# Scatter Plot
st.subheader("Area vs Price Graph")

fig, ax = plt.subplots()

ax.scatter(data['Area_sqft'], data['Price_INR'])
ax.set_xlabel("Area_sqft")
ax.set_ylabel("Price_INR")
ax.set_title("House Area vs Price")

st.pyplot(fig)

