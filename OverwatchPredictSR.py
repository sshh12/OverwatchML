
# coding: utf-8

# In[1]:

# Imports

from OverwatchProcessData import get_competitive_rank, get_vector_gamestats
from OverwatchGatherData import Player

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt

import numpy as np
import os

np.random.seed(5)


# In[2]:

# Load Data

def load_data():

    unscaled_X, unscaled_y = [], []

    for filename in os.listdir('profiles'):

        player = Player.from_file(os.path.join('profiles', filename))

        rank = get_competitive_rank(player, 'us')

        if rank: # Only use data w/rank attached

            unscaled_X.append(get_vector_gamestats(player, 'us', 'competitive'))
            unscaled_y.append(rank)

    unscaled_X = np.array(unscaled_X, dtype=np.float64)
    unscaled_y = np.array(unscaled_y, dtype=np.float64)
    
    return unscaled_X, unscaled_y


# In[3]:

# Standardize Data

def scale_data(unscaled_X, unscaled_y):
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler(with_mean=False) # SR is 1-5000 so scaling w/mean has weird effects

    X = scaler_X.fit_transform(unscaled_X)
    y = np.squeeze(scaler_y.fit_transform(unscaled_y.reshape(-1, 1)))
    
    return X, y, scaler_X, scaler_y


# In[4]:

# Keras Model

def get_model():

    model = Sequential()
    model.add(Dense(50, input_dim=68, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=68, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam') # MSE loss b/c regression
    
    return model


# In[5]:

# Learning function

def fit_to_data(model, *args, **kwargs): # Wrapper for keras model.fit( ... )

    history = model.fit(*args, **kwargs, shuffle=True, verbose=0)
    
    return history


# In[6]:

# Predict SR

def predict_sr(model, player):
    
    stats_vector = np.array([get_vector_gamestats(player, 'us', 'competitive')])
    X = scaler_X.transform(stats_vector)

    y_matrix = model.predict(X)
    sr = np.squeeze(scaler_y.inverse_transform(y_matrix))
    
    return int(sr)


# In[7]:

# Loads and trains model

X, y, scaler_X, scaler_y = scale_data(*load_data())

model = get_model()

history = fit_to_data(model, X, y, epochs=200, batch_size=128, validation_split=.10)

# Plot loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

