
from flask import Flask, render_template, jsonify

import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<battletag>')
def predict(battletag=''):

    response = jsonify({'battletag': battletag})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
