from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle  # Import the pickle library to load the model

app = Flask(__name__)

# Load the model from the pickle file
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define routes
@app.route('/')
def home():
    return render_template('index.html', prediction_text="")



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the HTML form
        testWeight = float(request.form['weight'])
        testHumidity = float(request.form['humidity'])
        testTemperature = float(request.form['temperature'])
        
        # Make prediction using the loaded model
        testPrediction = model.predict([[testWeight, testHumidity, testTemperature]])
        
        return render_template('index.html', prediction_text=f"Air Quality: {testPrediction}")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
