
# coding: utf-8

# In[1]:

# Imports

import numpy as np
import os


# In[2]:

# Player Data Structure

gametype = ['competitive', 'quickplay']

general_stats = "final_blows_avg_per_10_min, offensive_assists_most_in_game, melee_final_blows_most_in_game, environmental_kills, solo_kills_avg_per_10_min, medals_gold, healing_done_most_in_game, final_blows, solo_kills, teleporter_pad_destroyed_most_in_game, of_teams_hero_damage, medals_bronze, time_spent_on_fire_most_in_game, hero_damage_done_most_in_game, turrets_destroyed, deaths, defensive_assists, games_tied, kill_streak_best, eliminations_avg_per_10_min, time_played, shield_generator_destroyed_most_in_game, medals, melee_percentage_of_final_blows, all_damage_done, all_damage_done_avg_per_10_min, melee_final_blows_avg_per_10_min, shield_generator_destroyed, objective_kills_most_in_game, barrier_damage_done_avg_per_10_min, of_teams_damage, objective_time, games_lost, time_spent_on_fire_avg_per_10_min, healing_done_avg_per_10_min, multikills, kpd, hero_damage_done, objective_time_most_in_game, objective_time_avg_per_10_min, defensive_assists_most_in_game, final_blows_most_in_game, games_won, deaths_avg_per_10_min, damage_blocked, solo_kills_most_in_game, recon_assists, melee_final_blows, recon_assists_most_in_game, cards, all_damage_done_most_in_game, multikill_best, environmental_kills_most_in_game, turrets_destroyed_most_in_game, objective_kills, medals_silver, games_played, weapon_accuracy, eliminations, offensive_assists, hero_damage_done_avg_per_10_min, eliminations_most_in_game, barrier_damage_done, time_spent_on_fire, environmental_deaths, teleporter_pads_destroyed, healing_done, objective_kills_avg_per_10_min".split(', ')
   


# In[3]:


def get_competitive_rank(player, region):
    
    try:

        return int(player.json[region]['stats']['competitive']['overall_stats']['comprank'])

    except (TypeError, KeyError):

        return False


# In[4]:


def get_vector_gamestats(player, region, gametype, stat_keys=None):
    
    if not stat_keys:
            
        stat_keys = general_stats
        
    try:

        stats = player.json[region]['stats'][gametype]['game_stats']
            
        vector = []
            
        for key in stat_keys:
                
            if "overwatchguid" not in key:
                    
                if key not in stats:
                        
                    vector.append(0)
                        
                else:
                
                    vector.append(stats[key])
                
        return np.array(vector)

    except (TypeError, KeyError):

        return False

