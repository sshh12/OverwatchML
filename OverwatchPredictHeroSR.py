
# coding: utf-8

# In[1]:

# Imports

from OverwatchProcessData import get_vector_herostats
from OverwatchProcessData import get_competitive_rank, hero_stats
from OverwatchGatherData import Player, find_usernames

import numpy as np
import os

np.random.seed(3)

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt


# In[2]:

# Loading Data

specific_stats = {}

for stat in hero_stats:
    
    hero, _, _ = stat.split(" ")
    
    if hero in specific_stats:
        
        specific_stats[hero].append(stat)
    
    else:
        
        specific_stats[hero] = [stat]

def generate_players():
    
    for filename in os.listdir('profiles'):
        
        player = Player.from_file(os.path.join('profiles', filename))
        
        if 'error' not in player.json:
            
            yield player

def load_data(hero):

    unscaled_X, unscaled_y = [], []

    for player in generate_players():

        rank = get_competitive_rank(player, 'us')

        if rank:
            
            try:
                
                time_played = player.json['us']['heroes']['stats']['competitive'][hero]['general_stats']['time_played']
                
                if time_played >= .5:
                
                    unscaled_X.append(get_vector_herostats(player, 'us', stat_keys=specific_stats[hero]))
                    unscaled_y.append(rank)
                
            except:
                
                pass
           

    unscaled_X = np.array(unscaled_X, dtype=np.float64)
    unscaled_y = np.array(unscaled_y, dtype=np.float64)
    
    print(unscaled_X.shape, unscaled_y.shape)
    
    return unscaled_X, unscaled_y

len(specific_stats), len(specific_stats['mercy'])


# In[3]:

# Scale

def scale_data(unscaled_X, unscaled_y):
    
    scaler_X = StandardScaler()

    X = scaler_X.fit_transform(unscaled_X)
    y = unscaled_y / 5000
    
    return X, y, scaler_X


# In[4]:

# Model

def get_model(hero):
    
    model = Sequential()
    
    model.add(Dense(24, input_dim=len(specific_stats[hero]), kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(24, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(24, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(24, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(24, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(24, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model


# In[5]:

# Train wrapper

def train_model(model, *args, **kwargs):

    history = model.fit(*args, **kwargs, shuffle=True, validation_split=.10, verbose=0, callbacks=[EarlyStopping(patience=25)])
    
    return history

def get_hero_model(hero, from_file=False):
    
    if from_file:
        
        model = load_model(os.path.join('models', '{}-sr.h5'.format(hero)))
        
        scaler_X = joblib.load(os.path.join('models', '{}-sr.pkl'.format(hero)))
        
        return None, model, scaler_X
    
    print('TRAINING ' + hero)
    
    X, y, scaler_X = scale_data(*load_data(hero))

    model = get_model(hero)

    history = train_model(model, X, y, epochs=1500, batch_size=256)

    model.save(os.path.join('models', '{}-sr.h5'.format(hero)))
    joblib.dump(scaler_X, os.path.join('models', '{}-sr.pkl'.format(hero)))
    
    return history, model, scaler_X


# In[6]:

# Predict

def predict_sr(model, player, scaler_for_X, hero):
    
    stats_vector = np.array([get_vector_herostats(player, 'us', stat_keys=specific_stats[hero])])
    
    X = scaler_for_X.transform(stats_vector)

    y_matrix = model.predict(X)
    
    sr = np.squeeze(y_matrix) * 5000
    
    return int(sr)


# In[7]:

# View

def view(history):
    
    plt.plot(np.log(history.history['loss']))
    plt.plot(np.log(history.history['val_loss']))
    plt.title('Model Loss')
    plt.ylabel('Log(loss)')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
    
    plt.plot(np.sqrt(history.history['loss']) * 5000)
    plt.plot(np.sqrt(history.history['val_loss']) * 5000)
    plt.title('Model Accuracy')
    plt.ylabel('Avg Accuracy')
    plt.xlabel('epoch')
    plt.ylim([0, 1250])
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# In[8]:

# Run

for hero in specific_stats:
    
    history, model, _ = get_hero_model(hero)
    
    plt.plot(np.sqrt(history.history['val_loss']) * 5000)
    
plt.title('Model Accuracy')
plt.ylabel('Avg Accuracy')
plt.xlabel('epoch')
plt.ylim([0, 1300])
plt.legend(list(specific_stats), loc='lower left')
plt.show()


# In[9]:

# Load models from disk

models = {}

for hero in specific_stats:
    
    models[hero] = get_hero_model(hero, from_file=True)


# In[10]:

# Predict using all viable models

def predict_all(player):
    
    sr_predictions = []
    time_played = []

    for hero in specific_stats:
        
        player_hero_stats = player.json['us']['heroes']['stats']['competitive']

        if hero in player_hero_stats and player_hero_stats[hero]['general_stats']['time_played'] >= .2:

            _, model, scaler = models[hero]
                
            sr_predictions.append(predict_sr(model, player, scaler, hero))
            time_played.append(player_hero_stats[hero]['general_stats']['time_played'])
                
    return int(np.average(sr_predictions, weights=time_played))


# In[ ]:

# Test

with open('test_names.txt', 'r') as test:

    for battletag in find_usernames(test.read()):
        
        player = Player.from_web_battletag(battletag)
        
        actual = get_competitive_rank(player, 'us')
        p = predict_all(player)
        
        print("{} is {}, predicted {}".format(battletag, actual, p))

