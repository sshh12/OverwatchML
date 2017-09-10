
# coding: utf-8

# In[1]:

# Imports

import urllib.parse
import requests
import random
import base64
import praw
import time
import json
import re
import os


# In[2]:

# Reddit Config

# reddit = praw.Reddit(client_id='', client_secret='', user_agent='')


# In[3]:

# Tools

def find_usernames(text): # Text -> list<Battletags>
    
    for battletag in re.findall('[^\s=,;"*>+#\'\\^]{3,12}#\d{4,5}', text):
        
        battletag = battletag.strip()
        
        yield battletag

def get_user_json(battletag): # Battletag -> JSON
    
    battletag = battletag.replace('#', '-')
    
    text = requests.get('https://owapi.net/api/v3/u/{}/blob'.format(urllib.parse.quote_plus(battletag)), 
                            headers={'User-Agent':'OWAgent'}).text
    
    if "ratelimited" in text:
        
        time.sleep(4)
        
        return get_user_json(battletag)
    
    else:
        
        return text

def save_profile(battletag): # Battletag -> file.json
    
    print('Saving: ' + battletag)
    
    os.makedirs('profiles', exist_ok=True)
    
    b64_name = str(base64.b64encode(battletag.encode('utf-8'), b'=-'))[2:-1]
    filename = b64_name + '.json'
    
    if filename not in os.listdir('profiles'):
        
        data = get_user_json(battletag)
        
        if "error" not in data:
            
            with open(os.path.join('profiles', filename), 'w') as profile:
                
                profile.write(data)
                
            print('Saved: ' + battletag)
            
        else:
            
            if "404" in data:
                
                with open(os.path.join('profiles', filename), 'w') as profile:
                
                    profile.write(data)
                    
                print('Saved404: ' + battletag)
            
            print(data)
        


# In[4]:

# Player Object

class Player(object):

    def __init__(self, json_string):

        self.json = json.loads(json_string)

    @classmethod
    def from_file(self, filename):

        with open(filename, 'r') as profile:

            return Player(profile.read())
        
    @classmethod
    def from_web_battletag(self, battletag):
        
        json = get_user_json(battletag)
        
        if "error" not in json:

            return Player(json)
        
        raise Exception('Battletag Error ' + json)

    def get_regions(self):

        regions_with_data = []

        for region in ['us', 'eu', 'kr']:

            if self.json[region]:

                regions_with_data.append(region)

        return regions_with_data


# In[5]:

# Reddit Data

subreddits = ['Overwatch', 'CompetitiveOverwatch', 
              'OverwatchUniversity', 'Overwatch_Memes', 
              'Overwatched', 'OverwatchCustomGames',
              'OverwatchCirclejerk', 'wholesomeoverwatch',
              'ImaginaryOverwatch', 'Overwatch_comics', 
              'OverwatchHeroConcepts', 'OverwatchLore', 
              'OverwatchLeague', 'AnaMains', 
              'HanaSong', 'luciomains', 
              'LucioRollouts', 'McCreeMains', 
              'MeiMains', 'SymmetraMains', 
              'WidowmakerMains', 'ZaryaMains', 
              'ZenyattaMains', 'Hearthstone']

def generate_reddit_text(subreddit): # Subreddit -> the content of as many comments possible
    
    for submission in reddit.subreddit(subreddit).hot(limit=None):
        
        submission.comments.replace_more(limit=0)
        
        for comment in submission.comments.list():
            
            yield comment.body
            
def dl_reddit(): # Gathers data from reddit

    for sub in subreddits:

        for data in generate_reddit_text(sub):

            for battletag in find_usernames(data):

                save_profile(battletag)


# In[6]:

# Reddit Spreedsheet Data
# Thanks to https://www.reddit.com/r/Overwatch/comments/3qqs44/official_find_friends_by_posting_your_battletag/

public_spreedsheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRDbeVQe3Fi0vmodOHx3JeHpRteqjoUPgppklYaWeZe81i_vm0oavsEQUjLaoVHJNLbMe2EwubkXdXm/pub?output=csv"

def get_spreedsheet_text(url): # Gets content of spreedsheet
    
    return requests.get(url).text

def dl_spreedsheet(): # Retrieves all 1000+ battletags from the spreedsheet
    
    data = get_spreedsheet_text(public_spreedsheet_url)
    
    for battletag in find_usernames(data):
        
        save_profile(battletag)


# In[7]:


ow_tracker_methods = ['MostEliminations', 'MostDamageDone', 
                      'Level', 'CompetitiveRank', 
                      'EliminationsPM', 'DamageDonePM', 
                      'HealingDonePM', 'FinalBlowsPM', 
                      'Kd','Kad', 'Wl', 'Kg', 
                      'SoloKills', 'ObjectiveKills', 
                      'FinalBlows', 'DamageDone', 
                      'Eliminations', 'EnvironmentalKills', 
                      'MultiKills', 'Deaths', 
                      'GamesPlayed', 'TimeSpentOnFire', 
                      'ObjectiveTime', 'TimePlayed']

def get_overwatchtracker_pages(method, pages=1000, random=False):
    
    for i in range(pages):
        
        page_num = i + 1
        
        if random:
            
            page_num = random.randrange(1, 1400)
        
        yield requests.get('https://overwatchtracker.com/leaderboards/pc/global/{}?page={}&mode=1'.format(method, page_num)).text

def dl_overwatchtracker():
        
    for method in ow_tracker_methods:
        
        try:

            for page in get_overwatchtracker_pages(method, pages=300, random=True):

                for battletag in find_usernames(page):

                    save_profile(battletag)
                    
        except Exception as e:
            
            print(e)


# In[8]:

# Run

# dl_spreedsheet()
# dl_reddit()
# dl_overwatchtracker()

