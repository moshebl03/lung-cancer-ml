
# Lung Cancer Prediction Project

## Overview
This project analyzes a lung cancer dataset to predict cancer presence using machine learning.

## Files
- `lung_cancer_analysis.ipynb`: Jupyter Notebook with full analysis.
- `cleaned_lung_cancer.csv`: Cleaned dataset.
- `model.pkl`: Trained ML model.
- `preprocessor.pkl`: Preprocessing transformer.
- `selector.pkl`: Feature selector.
- `app.py`: Flask app for predictions.

## Setup
1. Install dependencies: `pip install pandas numpy scikit-learn flask matplotlib seaborn`
2. Place `lung_cancer.csv` in the project directory.
3. Run the notebook to generate outputs.
4. Run the Flask app: `python app.py`

## API Usage
Send a POST request to `http://localhost:5000/predict` with JSON data:
```json
{
  "age": 60,
  "gender": "Male",
  "race": "White",
  "smoker": "Current",
  "days_to_cancer": 500
}
```
Response: `{"has_cancer": 1}`
