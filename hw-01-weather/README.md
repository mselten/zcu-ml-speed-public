# Homework 01 - Weather prediction
Your task is to predict the weather based on meteorological data. The task is classification, you are predicting one of 12 classes.

## Data
I generated this data using Meteostat python library [link](https://dev.meteostat.net/python/). \
The training data has 12 columns:

| **Column** | **Description**                                                                     | **Type**   |
|:-----------|:------------------------------------------------------------------------------------|:-----------|
| time       | The datetime of the observation                                                     | Datetime64 |
| temp       | The air temperature in _°C_                                                         | Float64    |
| dwpt       | The dew point in _°C_                                                               | Float64    |
| rhum       | The relative humidity in percent (_%_)                                              | Float64    |
| prcp       | The one hour precipitation total in _mm_                                            | Float64    |
| snow       | The snow depth in _mm_                                                              | Float64    |
| wdir       | The average wind direction in degrees (_°_)                                         | Float64    |
| wspd       | The average wind speed in _km/h_                                                    | Float64    |
| wpgt       | The peak wind gust in _km/h_                                                        | Float64    |
| pres       | The average sea-level air pressure in _hPa_                                         | Float64    |
| tsun       | The one hour sunshine total in minutes (_m_)                                        | Float64    |
| coco       | This is the target that we shall predict                                            | Int        |

### Coco - target
The column `coco` is integer that represents weather code. The classes are:

| **Code** | **Weather**   |
|:---------|:--------------|
| 1        | Clear         |
| 2        | Cloudy        |
| 3        | Fog           |
| 4        | Rain          |
| 5        | Freezing Rain |
| 6        | Sleet         |
| 7        | Snowfall      |
| 8        | Rain Shower   |
| 9        | Snow Shower   |
| 10       | Storm         |
| 11       | Hail          |

The classes are not original from Meteostat library, but modified by me. The original classes were far too granular and I reduced them a bit.

## Files
In the file `data/train-data.csv` you will find the training data with targets, on which you should train your model. \
In the file `data/test-data.csv` you will find the test data without targets on which you should predict the weather.   

It is expected that the prediction file shall be `data/model-predictions.csv` and it shall have format (integers divided by newline):
```
2
4
6
...
3
2
```

After generating `model-predictions.csv` you can check the correct formating by running `validate-predictions.py` script.

## Submission
This homework will have format as competition and the model with the best accuracy will win. 

To submit your solution, please send me the `model-predictions.csv` file and the python script that you used to generate the predictions, so I can see approach you took.
Send it to my email that I will tell you in the class.

## Advice 
Start by dividing the training data into training and validation set. You can use the validation set to see true accuracy of your model. \
I recommend to start by training some baseline model. Logistic regression is your friend [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). \
After you have some baseline model, you can try to improve it by using more complex models (NN, Decision Trees, ...) \
I gave you fairly raw data, it might be good idea to do some preprocessing and feature engineering. \
When doing final prediction on test data, don't forget to train you data on all the training data you have including validation set. 

It shall be fairly easy to achieve accuracy of >50% with simple models. \
If you get stuck or have any questions, write me an email! One email can save you hours of frustration. 

Good luck! Imagination is the only limitation.


## Bonus
Normally it is forbidden to use data for training other than the ones given to you. But for fun you can use any data you can find. \
The script `BONUS-how-I-got-data.py` describes how I generated the dataset. \
The data is generated from some meteostation in central Romania. \
More data => better model?





