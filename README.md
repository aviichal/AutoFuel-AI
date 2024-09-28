
# AutoFuelAI

AutoFuelAI is a machine learning project designed to predict fuel efficiency in the automotive sector. By leveraging the 'Auto MPG' dataset and applying various regression models, the project aims to optimize predictions related to fuel consumption, contributing to sustainability in the automotive industry.

AutoFuelAI utilizes machine learning techniques to predict the miles per gallon (MPG) of vehicles. Through extensive data preprocessing, feature engineering, and model selection, the project enhances decision-making processes in the automotive industry for improved fuel efficiency.


## Features

- Data Cleaning & Feature Engineering: Handled missing values, cleaned and transformed categorical features.
- Exploratory Data Analysis (EDA): Performed detailed visualizations for feature analysis.
- Model Training: Implemented and compared various regression models such as Random Forest, Support Vector Regression, and others.
- Hyperparameter Tuning: Used GridSearchCV to optimize models for better predictions.
- Model Evaluation: Assessed models based on RMSE and R² scores.

## Dataset

The project uses the 'Auto MPG' dataset, which contains various features about vehicles, including:

- MPG (Miles Per Gallon): The target variable (fuel efficiency).
- Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, Origin: Predictive features for vehicle efficiency.
- Brand: Engineered from the vehicle's name for enhanced analysis.

The dataset is available at UCI Machine Learning Repository - Auto MPG.
## Exploratory Data Analysis

The project performs various forms of exploratory data analysis:

- Histograms & Box Plots: For understanding distribution and outliers in numeric features.
- Scatter Plots: To analyze the relationships between MPG, horsepower, weight, and displacement.
- Correlation Heatmap: For examining correlations between numeric features.
## Modeling and Evaluation

AutoFuelAI compares seven machine learning models to predict MPG:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- K-Nearest Neighbors (KNN) Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression (SVR)

The models are evaluated based on:

- Root Mean Squared Error (RMSE)
- R² Score

Best Model:

- Random Forest Regression: Achieved the best performance with an RMSE of 0.09 and an R² score of 0.92.
## RESULTS

The Random Forest Regression model gave the best results with:

- RMSE: 0.09
- R²: 0.92

This suggests that the model can predict fuel efficiency with high accuracy, making it suitable for practical automotive applications.
## Technologies Used

Programming Language: Python

Libraries:

- NumPy: For numerical operations.
- Pandas: For data manipulation and analysis.
- Matplotlib & Plotly: For visualizations.
- Scikit-learn: For implementing machine learning models and evaluation metrics.
- Seaborn: For statistical data visualization.