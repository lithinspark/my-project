import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
p=['blackgram','banana','mango','grapes','watermelon','muskmelon','apple','orange','papaya','coconut','cotton','jute','coffee']
p.sort()
print("AA ",p)
# Load the trained model
with open("rfmodel.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Convert to NumPy array
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction=f"Predicted Value: {prediction}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
