import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# Loading the model
model = pickle.load(open('Cement_Strength_Model.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('File.html')   

@app.route('/predict_api',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=float(model.predict(final_input)[0])
    return render_template("File.html",prediction_text="The Compressive strength of the concrete is {} MPa".format(round(output,2)))


if __name__=="__main__":
    app.run(debug=True)