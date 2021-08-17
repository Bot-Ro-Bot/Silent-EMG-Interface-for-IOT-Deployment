from flask import Flask

app = Flask(__name__)

@app.route("/predict")
def predict():
    pass


if __name__ == "__main__":
    app.run(debug=False)