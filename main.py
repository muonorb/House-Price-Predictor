import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('House Price Prediction/best_random_forest_model.pkl')

st.title('House Price Prediction')
st.write('Enter the details below to get the predicted house price.')

longitude = st.number_input('Longitude', -180.0, 180.0)
latitude = st.number_input('Latitude', -90.0, 90.0)
housing_median_age = st.number_input('Housing Median Age', 0, 100)
total_bedrooms = st.number_input('Total Bedrooms', 0, 10000)
population = st.number_input('Population', 0, 1000000)
households = st.number_input('Households', 0, 10000)
median_income = st.number_input('Median Income', 0.0, 30.0)
ocean_proximity = st.selectbox('Ocean Proximity', ['NEAR BAY', 'NEAR OCEAN', 'ISLAND', '<1H OCEAN', 'INLAND'])

pred_data = np.array([[longitude, latitude, housing_median_age, total_bedrooms, population, households, median_income]])

if ocean_proximity == 'NEAR BAY':
    append_data = [1,0,0,0,0]
elif ocean_proximity == 'NEAR OCEAN':
    append_data = [0,1,0,0,0]
elif ocean_proximity == 'ISLAND':
    append_data = [0,0,1,0,0]
elif ocean_proximity == '<1H OCEAN':
    append_data = [0,0,0,1,0]
elif ocean_proximity == 'INLAND':
    append_data = [0,0,0,0,1]
else:
    append_data = [0,0,0,0,0]


pred_data = np.append(pred_data, append_data)

pred_data = np.array([pred_data])
print(pred_data)

if st.button('Predict'):
    prediction = model.predict(pred_data)
    predicted_value = prediction[0]  
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; border: 2px solid #4c6ef5;">
                <h2 style="color: #4c6ef5; text-align: center; font-weight: bold;">
                    Predicted Median House Value: <br> ${predicted_value:,.2f}
                </h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

