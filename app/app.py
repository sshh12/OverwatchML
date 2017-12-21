
from flask import Flask, render_template, jsonify
import urllib.parse
import requests
import json

from predict import predict_all

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<battletag>')
def predict(battletag=''):

    api_resp = requests.get('https://owapi.net/api/v3/u/{}/blob'.format(urllib.parse.quote_plus(battletag)), headers={'User-Agent':'OWAgent'}).text
    api_json = json.loads(api_resp)

    overall_sr, (sr_predictions, heros, time_played) = predict_all(api_json)

    response = jsonify({
        'battletag': battletag,
        'api': api_json,
        'actualrank': int(api_json['us']['stats']['competitive']['overall_stats']['comprank']),
        'predictedrank': overall_sr,
        'specifics': {
            'sr': sr_predictions,
            'heros': heros,
            'timeplayed': time_played
        }
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
