from flask import Flask,jsonify,render_template,request
import pickle 
import numpy as np

app=Flask(__name__)
with open('customer_seg_model.pkl','rb')as f:
    model=pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature=[float(x) for x in request.form.values()]
    array=[np.array(feature)]
    pred=model.predict(array)[0]
    return render_template("home.html", prediction_text = "Category is {}".format(pred))

if __name__ == "__main__":
    app.run(port='3232',debug=True)