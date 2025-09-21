# ML Project 1 - Student Performance Prediction

Implementation of One ML Project end to end with use of Flask deployments and also basics of AWS.

## Project Overview:

The Student Performance Prediction project is focused on predicting student performance based on various features such as study habits, previous academic scores, and demographic data. The project leverages machine learning to analyze student data and predict their performance in subjects like math, reading, and writing. The overall aim is to assist educators and administrators in making data-driven decisions that can improve student outcomes.

## Key Techniques and Tools Used:

Python: The primary programming language used to implement the entire machine learning pipeline, from data ingestion to deployment.

Pandas: Utilized for data manipulation, cleaning, and preparation.

Scikit-learn: Used for building machine learning models, performing data preprocessing, and splitting datasets.

CatBoost, XGBoost, RandomForest, GradientBoosting: These machine learning models were used for prediction and regression tasks.

Flask: Deployed the trained model and preprocessing pipeline into a simple web application for real-time predictions.

Logging and Custom Exceptions: Employed for better debugging and error handling throughout the codebase.

## Steps Taken in the Project:

1. Data Ingestion:

The DataIngestion class is responsible for loading the student data from a CSV file (stud.csv). The class then splits the data into training and testing sets, saving them to separate CSV files.

File Handling: The code handles reading data from a file and ensures directories for saving data exist.

Train-Test Split: It uses train_test_split from scikit-learn to divide the data into training and test sets (80% training and 20% test).

Error Handling: Custom exceptions are raised in case of errors during ingestion.

2. Data Transformation:

The DataTransformation class is responsible for transforming the raw data into a format suitable for machine learning.

Column Transformation: Categorical and numerical columns are processed separately. Categorical columns are encoded using OneHotEncoder, and numerical columns are scaled using StandardScaler. Missing values are handled using SimpleImputer.

Preprocessing Pipelines: A ColumnTransformer is used to apply different transformations to different types of data.

Saving Preprocessor: After transforming the data, the preprocessing object is saved using the save_object function, which allows the preprocessing pipeline to be reused during prediction.

3. Model Training:

The ModelTrainer class is responsible for training various regression models on the data.

Model Selection: Several models are tested, including RandomForest, DecisionTree, GradientBoosting, XGBoost, CatBoost, and AdaBoost.

Hyperparameter Tuning: Hyperparameters for each model are tuned using a dictionary of parameters passed to the evaluate_models function.

Model Evaluation: After training, the r2_score is computed to evaluate model performance. The best model (with an R² score greater than 0.6) is selected and saved.

Saving Trained Model: The best-performing model is saved using the save_object function for future predictions.

4. Prediction Pipeline:

The PredictPipeline class provides a way to make predictions using the trained model and the preprocessing pipeline.

Model Loading: It loads the saved model and preprocessing pipeline from disk using the load_object function.

Transformation & Prediction: The input features are transformed using the preprocessor, and then the model predicts the student’s performance.

Additionally, the CustomData class is used to create custom inputs for making predictions. This class converts the user’s input (such as reading and writing scores) into a pandas DataFrame, which is then passed into the prediction pipeline.

## Challenges Faced and Overcome:

Data Handling: One common issue was ensuring the data was in the correct format for training the model. This involved proper handling of missing values, categorical variables, and scaling.

Model Tuning: Tuning the hyperparameters of the various models to get the best performance was challenging, especially with multiple models being tested. This was resolved by using cross-validation and selecting the model with the highest R² score.

Deployment: Deploying the model with Flask involved ensuring the prediction pipeline was ready to handle real-time input and output efficiently. Using the load_object and save_object methods streamlined this process.

## Results Achieved:

Model Performance: The models were evaluated based on R² score and the best model was selected. A model with an R² score above 0.6 was considered a success, showing a good level of accuracy for predicting student performance.

Deployment: The project was successfully deployed using Flask, allowing for real-time predictions via a web interface. The user can input new student data (e.g., reading and writing scores), and the model will predict the student's performance.
