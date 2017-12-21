from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.models import load_model
import numpy as np
import stats
import os

import keras.backend as K
import keras.metrics
keras.metrics.acc_metric = lambda a, b: K.mean(K.abs(b - a) < .05, axis=-1)

## Options
MODEL_PATH = os.path.join('..', 'models')

## Load stat names

specific_stats = {}

for stat in stats.hero_stats:

    hero, _, _ = stat.split(" ")

    if hero in specific_stats:

        specific_stats[hero].append(stat)

    else:

        specific_stats[hero] = [stat]

## Load models

models = {}

for hero in specific_stats:

    print('Loading ' + hero.title())

    model = load_model(os.path.join(MODEL_PATH, '{}-sr.h5'.format(hero)))
    scaler_X = joblib.load(os.path.join(MODEL_PATH, '{}-sr.pkl'.format(hero)))

    models[hero] = (model, scaler_X)

## Encoder JSON profile into vector

def get_vector_herostats(player_json, stat_keys=None):

    if not stat_keys:

        stat_keys = stats.hero_stats

    stats = player_json['us']['heroes']['stats']['competitive']

    vector = []

    for stat in stat_keys:

        hero, stat_type, value = stat.split(" ")

        try:

            vector.append(stats[hero][stat_type][value])

        except (TypeError, KeyError):

            vector.append(0)

    return np.array(vector)

## Predict for a single hero

def predict_sr(player_json, hero):

    model, scalerX = models[hero]

    stats_vector = np.array([get_vector_herostats(player_json, stat_keys=specific_stats[hero])])

    X = scalerX.transform(stats_vector)

    y_predict = model.predict(X)

    sr = np.squeeze(y_predict) * 5000

    return int(sr)

## Predict with all heros

def predict_all(player_json, min_time_played=.25):

    sr_predictions, heros, time_played = [], [], []

    try:

        player_hero_stats = player_json['us']['heroes']['stats']['competitive']

    except (TypeError, KeyError):

        return -1, ([], [], [])

    for hero in specific_stats:

        if hero in player_hero_stats and player_hero_stats[hero]['general_stats']['time_played'] >= min_time_played:

            sr_predictions.append(predict_sr(player_json, hero))
            heros.append(hero.title())
            time_played.append(player_hero_stats[hero]['general_stats']['time_played'])

    overall_sr = int(np.average(sr_predictions, weights=time_played)) # Weighted average of sr by time played

    return overall_sr, (sr_predictions, heros, time_played)
