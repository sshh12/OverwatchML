
from flask import Flask, render_template, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.models import load_model
import urllib.parse
import requests
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<battletag>')
def predict(battletag=''):

    api_resp = requests.get('https://owapi.net/api/v3/u/{}/blob'.format(urllib.parse.quote_plus(battletag)), headers={'User-Agent':'OWAgent'}).text
    api_json = json.loads(api_resp)

    response = jsonify({
        'battletag': battletag,
        'api': api_json,
        'actualrank': int(api_json['us']['stats']['competitive']['overall_stats']['comprank']),
        'predictedrank': 0
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
