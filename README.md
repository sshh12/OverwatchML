# OverwatchML

### Predicting SR

The goal of this project is to use player statistics ingame to predict their SR (Skill rating).

## App

Applying the Model

TODO

## Lab

Creating/Training Models

##### Gathering Data

A simple web scraper was used to extract battletags from reddit and [overwatchtracker](https://overwatchtracker.com/leaderboards/pc/global). The battletags
were then sent through [OWAPI](https://github.com/SunDwarf/OWAPI/blob/master/api.md) to retrieve the stats
in an easy to work with json.

[View](https://github.com/sshh12/OverwatchML/blob/master/lab/OverwatchGatherData.ipynb)

##### Processing

The pretrain data processing is pretty straightforward. Various methods extract their own combination
of values from the player json to test the effect of different features.

[View](https://github.com/sshh12/OverwatchML/blob/master/lab/OverwatchProcessData.ipynb)

##### Training and Predicting

A variety of mlp models are created using [Keras](https://keras.io/) and each are trained on their own dataset created from the processing step after being scaled to the same mean and deviation.

[View](https://github.com/sshh12/OverwatchML/blob/master/lab/OverwatchPredictSR.ipynb)

After seeing this [reddit post](https://www.reddit.com/r/Overwatch/comments/6vcoex/i_used_deep_learning_to_guess_your_sr_estimate/) I tried the one-trick idea with a model trained for each hero.

[View](https://github.com/sshh12/OverwatchML/blob/master/lab/OverwatchPredictHeroSR.ipynb)