# OverwatchML

### Predicting SR

The goal of this project is to use player statistics ingame to predict their SR (Skill rating).

##### Gathering Data

A simple web scraper was used to extract battletags from reddit. The battletags
were then sent through [OWAPI](https://github.com/SunDwarf/OWAPI/blob/master/api.md) to retrieve the stats
in an easy to work with json.

##### Processing

The pretrain data processing is pretty straightforward. Various methods extract their own combination
of values from the player json and the returned vectors are standardized to have a mean of 0 and a standard deviation of 1.

##### Training and Predicting

A variety of mlp models are created using [Keras](https://keras.io/) and each are trained on their own dataset created from the processing step.