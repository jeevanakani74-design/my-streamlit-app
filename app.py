import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("Titanic Linear Regression App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Titanic CSV File", type=["csv"])

if uploaded_file is not None:

    # Read CSV
    sps = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(sps.head())

    st.subheader("Dataset Information")
    st.write(sps.dtypes)

    # Check required columns
    if 'age' in sps.columns and 'parch' in sps.columns:

        # Remove missing values
        data = sps[['age', 'parch']].dropna()

        # Independent and dependent variables
        inde = data[['age']]
        dep = data['parch']

        # Train model
        LR = LinearRegression()
        LR.fit(inde, dep)

        st.success("Model Trained Successfully")

        # User input
        age_input = st.number_input(
            "Enter Age",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

        # Prediction button
        if st.button("Predict Parch"):
            prediction = LR.predict([[age_input]])
            st.subheader("Prediction Result")
            st.write(f"Predicted Parch Value: {prediction[0]:.2f}")

    else:
        st.error("CSV file must contain 'age' and 'parch' columns")
