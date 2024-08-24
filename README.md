# House Price Prediction using Machine learning

## Overview

This project focuses on predicting housing prices using machine learning techniques. By leveraging popular Python libraries such as NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, and Random Forest, this project provides an end-to-end solution for accurate price estimation.

## Features

- **Machine Learning Model:** Implements a Random Forest Regressor optimized through GridSearchCV for accurate predictions.
- **User Interface:** The model is integrated into a user-friendly web application using Streamlit, allowing users to input property features and get real-time price predictions.
- **Data Visualization:** Visualizes data insights and predictions using Matplotlib and Seaborn.

## Libraries Used

- **NumPy:** For numerical operations.
- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For building and evaluating the Random Forest model.
- **Matplotlib & Seaborn:** For data visualization.
- **Streamlit:** For building an interactive web application.

## Project Structure

- **model.py:** Contains the code for loading and training the Random Forest model.
- **app.py:** Streamlit code for the web interface, allowing users to enter features and see predicted house prices.
- **housing.csv:** The dataset used for training the model.

## Installation

1. Clone this repository:

   git clone https://github.com/muonorb/House-Price-Predictor.git

2. Navigate to the project directory:

   cd House-Price-Predictor

3. Install the required libraries:

   pip install -r requirements.txt

## Usage

1. Run the Streamlit application:

   streamlit run main.py

2. Open the provided URL in your web browser to access the interface.

3. Enter the required property details and get the predicted house price instantly.

## Model Details

The model is built using a **Random Forest Regressor** which was fine-tuned using GridSearchCV to select the best hyperparameters. The model was trained on a dataset containing various features like location (latitude and longitude), housing median age, number of bedrooms, population, and proximity to the ocean.

## Results

The project achieved accurate predictions with the Random Forest model, making it an effective tool for estimating house prices.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any improvements.

## License

This project is licensed under the MIT License.
