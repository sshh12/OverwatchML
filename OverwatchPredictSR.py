
# coding: utf-8

# In[1]:

# Imports

from OverwatchProcessData import get_vector_combined, get_vector_gamestats, get_vector_herostats
from OverwatchProcessData import get_competitive_rank, general_stats, hero_stats
from OverwatchGatherData import Player, find_usernames

import numpy as np
import os

np.random.seed(3333)

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt


# In[2]:

# Load Data

## Creating Custom Metrics

general_general_stats = ['kpd']

for stat in general_stats:

    if 'avg' in stat:

        general_general_stats.append(stat)
        
general_hero_stats = []

for stat in hero_stats:

    if 'avg' in stat:

        general_hero_stats.append(stat)
        
## Loading Data
        
def generate_players():
    
    for filename in os.listdir('profiles'):
        
        player = Player.from_file(os.path.join('profiles', filename))
        
        if 'error' not in player.json:
            
            yield player

def load_data(get_vector):

    unscaled_X, unscaled_y = [], []

    for player in generate_players():

        rank = get_competitive_rank(player, 'us')

        if rank:
            
            playtime = player.json['us']['stats']['competitive']['game_stats']['time_played']
            
            if playtime > 20:

                unscaled_X.append(get_vector(player, 'us'))
                unscaled_y.append(rank)

    unscaled_X = np.array(unscaled_X, dtype=np.float64)
    unscaled_y = np.array(unscaled_y, dtype=np.float64)
    
    print(unscaled_X.shape)
    print(unscaled_y.shape)
    
    return unscaled_X, unscaled_y


# In[3]:

# Standardize Data

def scale_data(unscaled_X, unscaled_y):
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler(with_mean=False) # SR is 1-5000 so scaling w/mean has weird effects

    X = scaler_X.fit_transform(unscaled_X)
    y = np.squeeze(scaler_y.fit_transform(unscaled_y.reshape(-1, 1)))
    
    return X, y, scaler_X, scaler_y

def scale_data2(unscaled_X, unscaled_y):
    
    scaler_X = StandardScaler()

    X = scaler_X.fit_transform(unscaled_X)
    y = unscaled_y / 5000
    
    return X, y, scaler_X


# In[4]:

# Keras Model

def get_model():

    model = Sequential()
    model.add(Dense(40, input_dim=len(general_stats), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam') # MSE loss b/c regression
    
    return model

def get_model2():
        
    model = Sequential()
    model.add(Dense(13, input_dim=len(general_general_stats), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(13, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

def get_model3():
        
    model = Sequential()
        
    model.add(Dense(24, input_dim=len(hero_stats), kernel_initializer='normal', activation='relu'))
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
    
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

def get_model4():
        
    model = Sequential()
    model.add(Dense(12, input_dim=len(general_hero_stats), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

def get_model5():
        
    model = Sequential()
    model.add(Dense(20, input_dim=3158, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

def get_model_from_file():
    
    model = load_model(os.path.join('models', 'overall-sr.h5'))
    scalar = joblib.load(os.path.join('models', 'overall-sr.pkl'))
    
    return model, scalar


# In[5]:

# Learning function. Wrapper for keras model.fit( ... )

def train_model(model, *args, **kwargs):
    
    # print(model.summary())

    history = model.fit(*args, **kwargs, shuffle=True, validation_split=.10, verbose=0, callbacks=[EarlyStopping(patience=30)])
    
    return history


# In[6]:

# Predict SR

def predict_sr(model, player):
    
    stats_vector = np.array([get_vector_gamestats(player, 'us')])
    
    X = scaler_X.transform(stats_vector)

    y_matrix = model.predict(X)
    
    sr = np.squeeze(scaler_y.inverse_transform(y_matrix))
    
    return int(sr)

def predict_sr2(model, player, scaler_for_X, get_vector):
    
    stats_vector = np.array([get_vector(player, 'us')])
    
    X = scaler_for_X.transform(stats_vector)

    y_matrix = model.predict(X)
    
    sr = np.squeeze(y_matrix) * 5000
    
    return int(sr)


# In[7]:

# Stats

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
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
    


# In[8]:

# Run (Load)

get_vector = lambda player, region : get_vector_herostats(player, region)

X, y, scaler_X = scale_data2(*load_data(get_vector))


# In[9]:

# Run (Train)

model = get_model3()

history = train_model(model, X, y, epochs=1500, batch_size=256)

model.save(os.path.join('models', 'overall-sr.h5'))
joblib.dump(scaler_X, os.path.join('models', 'overall-sr.pkl'))

view(history)


# In[10]:

# Run (Load from disk)

get_vector = lambda player, region : get_vector_herostats(player, region)

model, scaler_X = get_model_from_file()


# In[ ]:

# Run (Test)

with open('test_names.txt', 'r') as test:

    for battletag in find_usernames(test.read()):
        
        player = Player.from_web_battletag(battletag)
        
        actual = get_competitive_rank(player, 'us')
        p = predict_sr2(model, player, scaler_X, get_vector)
        
        print('{} is {}, predicted {}'.format(battletag, actual, p))

