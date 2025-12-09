# Student Placement Prediction Web App

This project predicts whether a student is likely to be placed or not placed based on multiple academic and skill-based features.
It includes:

 - Exploratory Data Analysis (EDA)

 - Data cleaning & feature selection

 - Three machine learning models (RF, LR, ANN)

 - A Flask web application that allows users to upload a CSV and receive predictions

## Machine Learning Models

This project evaluates three different machine learning models, all trained on the same cleaned dataset.
The goal was to compare their accuracy, stability, and real-world performance.

### Models Implemented
* Model1.py — Random Forest Classifier: A robust ensemble model that handles nonlinear relationships well and performs consistently.
* Model2.py — Logistic Regression: A simple, interpretable baseline model used for comparison.
* Model3.py — Artificial Neural Network (ANN): A multi-layer perceptron model capable of learning nonlinear patterns.

Only Model 1 (Random Forest) is used by the web interface because it shows the best accuracy and stability.

## EDA & Data Cleaning (eda.ipynb)
The notebook performs:

1. Dataset loading and basic inspection

    - Reads college_student_placement_dataset.csv

    - Displays column types and missing values

2. Removal of irrelevant features (College_ID removed)

    - It is just an identifier

    - Adds noise

    - Does NOT represent student ability

3. Correlation Heatmap Analysis (Identified strong correlation)

    To avoid redundancy:

    - Removed CGPA

    - Kept Prev_Sem_Result (more meaningful for recent academic performance)

4. Final cleaned dataset saved as cleaned.csv

## Web Application (app.py)

The web app allows users to:

1. Upload a CSV file with the following columns:

    - Prev_Sem_Result
    
    - Academic_Performance
    
    - Internship_Experience
    
    - Extra_Curricular_Score
    
    - Communication_Skills
    
    - Projects_Completed
    
2. The file is scaled using the same StandardScaler as the model.

3. Predictions are generated using Random Forest.

4. Results are displayed in an HTML table.

5. Result will had extra column ' Predicted_Placement '

## How to Run the Web App

1. Create and activate a virtual environment
    <pre> python -m venv venv 

    .\venv\Scripts\Activate.ps1 </pre>

2. Install project dependencies
    <pre> pip install -r requirements.txt </pre>

3. Run the Flask server
    <pre> python app.py </pre>

4. Open the web app in your browser
    <pre> http://127.0.0.1:5000/ </pre>

## Summary

This project demonstrates a full ML pipeline:

- Dataset cleaning & EDA  
- Feature selection using correlation analysis  
- Multiple machine learning models compared  
- Fully functional Flask app  
- Clean UI for CSV upload and prediction  

## Future Improvements

There are several directions to improve and extend this project in the future:

### 1. Enhanced User Interface
- Add a cleaner dashboard-style layout with cards and icons.
- Display summary statistics (e.g., number of students placed / not placed) after each upload.
- Allow users to sort and filter the prediction table (e.g., show only “Not Placed” students).

### 2. Better Result Visualization
- Plot bar charts of feature distributions for the uploaded dataset.
- Show feature importance for the Random Forest model.
- Add a comparison view to show how a particular student’s features differ from the overall average.



