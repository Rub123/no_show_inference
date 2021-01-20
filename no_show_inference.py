from flask import Flask
from flask import request
import pandas as pd
import pickle
import json
import os
from preprocess_data import preprocess
# import sklearn

app = Flask(__name__)

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

    
@app.route('/')
def index():
    return 'Welcome to No-Show Inference Server'
    
    
@app.route('/predict', methods=["POST"])
def predict():
    if not request.is_json:
        return "Not a Valid Request", 400

    X = pd.DataFrame(json.loads(request.get_json()))
    X = preprocess(X)
    y_pred = model.predict_proba(X)[:, 1]
    return json.dumps(list(y_pred))


if __name__ == '__main__':
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()

