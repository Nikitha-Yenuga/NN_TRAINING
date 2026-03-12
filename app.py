from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("models/iris_model.h5")

# Iris class labels
classes = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(data)
    result = classes[np.argmax(prediction)]

    return render_template("index.html", prediction=result)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
