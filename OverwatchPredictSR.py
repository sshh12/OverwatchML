
# coding: utf-8

# In[1]:

# Imports

from OverwatchProcessData import get_competitive_rank, get_vector_gamestats, get_vector_herostats, get_vector_combined, general_stats, hero_stats
from OverwatchGatherData import Player, find_usernames

from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model

import matplotlib.pyplot as plt

import numpy as np
import os

np.random.seed(5)


# In[2]:

# Load Data

## Creating Custom Metrics

general_general_stats = ["kpd"]

for stat in general_stats:

    if "avg" in stat:

        general_general_stats.append(stat)
        
general_hero_stats = []

for stat in hero_stats:

    if "avg" in stat:

        general_hero_stats.append(stat)
        
## Loading Data
        
def generate_players():
    
    for filename in os.listdir('profiles'):
        
        yield Player.from_file(os.path.join('profiles', filename))

def load_data():

    unscaled_X, unscaled_y = [], []

    for player in generate_players():

        rank = get_competitive_rank(player, 'us')
        rank2 = get_competitive_rank(player, 'eu')

        if rank: # Only use data w/rank attached

            unscaled_X.append(get_vector_gamestats(player, 'us'))
            unscaled_y.append(rank)
            
        if rank2:

            unscaled_X.append(get_vector_gamestats(player, 'eu'))
            unscaled_y.append(rank2)

    unscaled_X = np.array(unscaled_X, dtype=np.float64)
    unscaled_y = np.array(unscaled_y, dtype=np.float64)
    
    return unscaled_X, unscaled_y

def load_data2():

    unscaled_X, unscaled_y = [], []

    for player in generate_players():

        rank = get_competitive_rank(player, 'us')
        rank2 = get_competitive_rank(player, 'eu')

        if rank:

            unscaled_X.append(get_vector_gamestats(player, 'us', stat_keys=general_general_stats))
            unscaled_y.append(rank)
            
        if rank2:
            
            unscaled_X.append(get_vector_gamestats(player, 'eu', stat_keys=general_general_stats))
            unscaled_y.append(rank2)

    unscaled_X = np.array(unscaled_X, dtype=np.float64)
    unscaled_y = np.array(unscaled_y, dtype=np.float64)
    
    return unscaled_X, unscaled_y

def load_data3():

    unscaled_X, unscaled_y = [], []

    for player in generate_players():

        rank = get_competitive_rank(player, 'us')
        rank2 = get_competitive_rank(player, 'eu')

        if rank:

            unscaled_X.append(get_vector_herostats(player, 'us'))
            unscaled_y.append(rank)
            
        if rank2:
            
            unscaled_X.append(get_vector_herostats(player, 'eu'))
            unscaled_y.append(rank2)

    unscaled_X = np.array(unscaled_X, dtype=np.float64)
    unscaled_y = np.array(unscaled_y, dtype=np.float64)
    
    return unscaled_X, unscaled_y

def load_data4():

    unscaled_X, unscaled_y = [], []

    for player in generate_players():

        rank = get_competitive_rank(player, 'us')
        rank2 = get_competitive_rank(player, 'eu')

        if rank:

            unscaled_X.append(get_vector_herostats(player, 'us', stat_keys=general_hero_stats))
            unscaled_y.append(rank)
            
        if rank2:
            
            unscaled_X.append(get_vector_herostats(player, 'eu', stat_keys=general_hero_stats))
            unscaled_y.append(rank2)

    unscaled_X = np.array(unscaled_X, dtype=np.float64)
    unscaled_y = np.array(unscaled_y, dtype=np.float64)
    
    return unscaled_X, unscaled_y

def load_data5():

    unscaled_X, unscaled_y = [], []

    for player in generate_players():

        rank = get_competitive_rank(player, 'us')
        rank2 = get_competitive_rank(player, 'eu')

        if rank:

            unscaled_X.append(get_vector_combined(player, 'us'))
            unscaled_y.append(rank)
            
        if rank2:
            
            unscaled_X.append(get_vector_combined(player, 'eu'))
            unscaled_y.append(rank2)

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

def scale_data2(unscaled_X, unscaled_y):
    
    scaler_X = StandardScaler()

    X = scaler_X.fit_transform(unscaled_X)
    y = unscaled_y / 5000
    
    return X, y, scaler_X


# In[4]:

# Keras Model

def get_model(from_file=False):
    
    if not from_file:

        model = Sequential()
        model.add(Dense(40, input_dim=68, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(40, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(40, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        model.compile(loss='mean_squared_error', optimizer='adam') # MSE loss b/c regression
        
    else:
        
        model = load_model(os.path.join('models', 'overwatch-sr-1.h5'))
    
    return model

def get_model2(from_file=False):
    
    if not from_file:
        
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(13, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(13, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        
    else:
        
        model = load_model(os.path.join('models', 'overwatch-sr-2.h5'))
    
    return model

def get_model3(from_file=False):
    
    if not from_file:
        
        model = Sequential()
        model.add(Dense(8, input_dim=3090, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        
    else:
        
        model = load_model(os.path.join('models', 'overwatch-sr-3.h5'))
    
    return model

def get_model4(from_file=False):
    
    if not from_file:
        
        model = Sequential()
        model.add(Dense(8, input_dim=410, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        
    else:
        
        model = load_model(os.path.join('models', 'overwatch-sr-4.h5'))
    
    return model

def get_model5(from_file=False):
    
    if not from_file:
        
        model = Sequential()
        model.add(Dense(20, input_dim=3158, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(20, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        
    else:
        
        model = load_model(os.path.join('models', 'overwatch-sr-5.h5'))
    
    return model


# In[5]:

# Learning function. Wrapper for keras model.fit( ... )

def train_model(model, *args, **kwargs):

    history = model.fit(*args, **kwargs, shuffle=True, verbose=0)
    
    return history


# In[6]:

# Predict SR

def predict_sr(model, player):
    
    stats_vector = np.array([get_vector_gamestats(player, 'us')])
    
    X = scaler_X.transform(stats_vector)

    y_matrix = model.predict(X)
    
    sr = np.squeeze(scaler_y.inverse_transform(y_matrix))
    
    return int(sr)

def predict_sr2(model, player):
    
    stats_vector = np.array([get_vector_gamestats(player, 'us', stat_keys=general_general_stats)])
    
    X = scaler_X2.transform(stats_vector)

    y_matrix = model.predict(X)
    
    sr = np.squeeze(y_matrix) * 5000
    
    return int(sr)

def predict_sr3(model, player):
    
    stats_vector = np.array([get_vector_herostats(player, 'us')])
    
    X = scaler_X3.transform(stats_vector)

    y_matrix = model.predict(X)
    
    sr = np.squeeze(y_matrix) * 5000
    
    return int(sr)

def predict_sr4(model, player):
    
    stats_vector = np.array([get_vector_herostats(player, 'us', stat_keys=general_hero_stats)])
    
    X = scaler_X4.transform(stats_vector)

    y_matrix = model.predict(X)
    
    sr = np.squeeze(y_matrix) * 5000
    
    return int(sr)

def predict_sr5(model, player):
    
    stats_vector = np.array([get_vector_combined(player, 'us')])
    
    X = scaler_X5.transform(stats_vector)

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
    


# In[8]:

# Model 1

X, y, scaler_X, scaler_y = scale_data(*load_data())

model = get_model()

history = train_model(model, X, y, epochs=400, batch_size=100, validation_split=.10)

model.save(os.path.join('models', 'overwatch-sr-1.h5'))

view(history)


# In[9]:

# Model 2

X, y, scaler_X2 = scale_data2(*load_data2())

model2 = get_model2()

history2 = train_model(model2, X, y, epochs=200, batch_size=100, validation_split=.10)

model2.save(os.path.join('models', 'overwatch-sr-2.h5'))

view(history2)


# In[10]:

# Model 3

X, y, scaler_X3 = scale_data2(*load_data3())

model3 = get_model3()

history3 = train_model(model3, X, y, epochs=500, batch_size=100, validation_split=.10)

model3.save(os.path.join('models', 'overwatch-sr-3.h5'))

view(history3)


# In[11]:

# Model 4

X, y, scaler_X4 = scale_data2(*load_data4())

model4 = get_model4()

history4 = train_model(model4, X, y, epochs=500, batch_size=100, validation_split=.10)

model4.save(os.path.join('models', 'overwatch-sr-4.h5'))

view(history4)


# In[12]:

# Model 5

X, y, scaler_X5 = scale_data2(*load_data5())

model5 = get_model5()

history5 = train_model(model5, X, y, epochs=500, batch_size=100, validation_split=.10)

model5.save(os.path.join('models', 'overwatch-sr-5.h5'))

view(history4)


# In[ ]:


with open('test_names.txt', 'r') as test:

    for battletag in find_usernames(test.read()):
        
        player = Player.from_web_battletag(battletag)
        
        actual = get_competitive_rank(player, 'us')
        p1 = predict_sr(model, player)
        p2 = predict_sr2(model2, player)
        p3 = predict_sr3(model3, player)
        p4 = predict_sr4(model4, player)
        p5 = predict_sr5(model5, player)
        avg = (p1 + p2 + p3 + p4 + p5) / 5.
        
        print("{} is {}, predicted {}, {}, {}, {}, {}, avg {}".format(battletag, actual, p1, p2, p3, p4, p5, avg))

