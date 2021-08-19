from inference import Inference
import random
import os

from flask import Flask,jsonify,request,render_template
# from werkzeug import secure_filename
# Create an application instance : app lekhne norm raixa
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload",methods=["GET","POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        # print("File received")
        data = "अबको समय सुनाउ"
        return render_template("index.html",data=data)


@app.route("/predict",methods=["POST"])
def predict():
    """
    predict function is an end point to this API to make a prediction and return it
    """
    # get an audio file from a client through POST request and save it
    signal = request.files["file"]
    filename = str(random.randint(0,1000))
    signal.save(filename)

    # making an inference of the model through singleton inference class
    inf = Inference()

    prediction = inf.predict(filename)
    os.remove(filename)

    # data= {"label":prediction}
    # return jsonify(data)
    return render_template("index.html",data=prediction)



if __name__ == "__main__":
    app.run(debug=True)