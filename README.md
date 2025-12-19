Car Price Prediction Web Application
Project Overview

This project is a machine learning based web application designed to predict the selling price of used cars. The application allows users to select different trained machine learning models, input car related features, and receive a predicted selling price in real time.

The system is built using Python and Streamlit, with multiple regression models trained using scikit learn. A modern user interface is implemented using custom CSS to provide a professional dashboard like experience.

The application is deployed on Streamlit Cloud and is accessible through a web browser.

Objectives

The primary objectives of this project are:

To build an accurate car price prediction system using machine learning techniques
To compare the performance of multiple regression models
To provide an interactive and user friendly web interface
To deploy the application as a cloud based web service

Machine Learning Models Used

The following machine learning models are trained and used in this project:

Linear Regression
Ridge Regression
Lasso Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor

Each model is evaluated based on accuracy, and users can select any model for prediction through the application interface.

Features

Model selection from a dropdown menu
Display of model accuracy for comparison
User input for car details such as year, price, kilometers driven, fuel type, seller type, transmission, and ownership
Real time prediction of car selling price
Professional dashboard style user interface
Fully deployed cloud based application

Technologies Used

Programming Language
Python

Libraries and Frameworks
Streamlit
Pandas
NumPy
Scikit learn
Pickle

Frontend Styling
Custom HTML and CSS integrated with Streamlit

Deployment Platform
Streamlit Cloud

Project Structure

The project follows a simple and clear structure:

app.py
requirements.txt
runtime.txt
saved_models directory containing trained model files

The saved_models directory stores the trained machine learning models in serialized format.

Installation and Local Setup

To run this project locally, follow these steps:

Clone the repository to your local machine
Create and activate a virtual environment
Install the required dependencies using the requirements file
Run the Streamlit application

The application will start on a local server and can be accessed through a web browser.

Deployment

The application is deployed using Streamlit Cloud.
All required dependencies are listed in the requirements file, and the Python version is controlled using the runtime file to ensure compatibility with scikit learn.

Usage Instructions

Open the deployed application in a web browser
Select a machine learning model from the sidebar
Enter the required car details in the input fields
Click the predict button
View the predicted selling price displayed on the screen

Results

The application successfully predicts car prices using multiple machine learning models.
Ensemble based models such as Random Forest and Gradient Boosting demonstrate higher accuracy compared to linear models.

Limitations

The prediction accuracy depends on the quality and size of the training dataset
The model is trained on historical data and may not reflect sudden market changes
Categorical values are limited to predefined options

Future Enhancements

Integration of more advanced models and hyperparameter tuning
Addition of more car features such as brand and engine capacity
Model retraining using larger and more recent datasets
Downloadable prediction reports
User authentication and history tracking

Conclusion

This project demonstrates the practical application of machine learning in predicting car prices through an interactive web application. It combines data science, machine learning, and web deployment to deliver a complete end to end solution suitable for real world usage and portfolio demonstration.
