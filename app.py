import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
# Set template_folder='.' to look for index.html in the same directory
app = Flask(__name__, template_folder='.')

# --- 1. Load Model and Preprocessing Artifacts ---
# These should be in the same directory as the app.
try:
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    selector = joblib.load('selector.pkl')
except FileNotFoundError as e:
    print(f"ERROR: A required model file was not found. Please ensure 'model.pkl', 'preprocessor.pkl', and 'selector.pkl' exist.")
    print(f"Error details: {e}")
    # Set to None to handle errors gracefully in the predict route
    model, preprocessor, selector = None, None, None

# --- 2. Define Options for the Frontend Form ---
# These are passed to the Jinja2 template in `index.html`
RACE_OPTIONS = [
    'White', 
    'Black or African-American',
    'Asian', 
    'American Indian or Alaskan Native',
    'Native Hawaiian or Other Pacific Islander', 
    'More than one race',
    'Unknown'
]

SMOKER_OPTIONS = [
    'Current', 
    'Former'
]

# --- 3. Define Application Routes ---

@app.route('/')
def home():
    """
    Renders the main user interface page (index.html).
    """
    return render_template(
        'index.html', 
        race_options=RACE_OPTIONS, 
        smoker_options=SMOKER_OPTIONS
    )

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the frontend's JavaScript.
    Receives patient data as JSON, preprocesses it using the loaded
    transformers, and returns the cancer risk prediction.
    """
    if not all([model, preprocessor, selector]):
        return jsonify({"error": "Model or preprocessors are not loaded. Server configuration error."}), 500

    # Get data from the POST request
    data = request.get_json()
    
    # --- Preprocessing ---
    # Convert the incoming JSON into a single-row DataFrame
    # The column names must match what the preprocessor was trained on
    try:
        df = pd.DataFrame([data])
        
        # Apply the same preprocessing steps from the notebook
        X_transformed = preprocessor.transform(df)
        X_selected = selector.transform(X_transformed)
        
        # Make the prediction (0 or 1)
        prediction = model.predict(X_selected)
        
        # Return the result as a JSON object
        return jsonify({'has_cancer': int(prediction[0])})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 400

# --- 4. Run the Application ---
if __name__ == '__main__':
    # Set debug=False for production
    app.run(debug=True)