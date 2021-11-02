# diabetes_prediction

## Problem Description

## Folder structure overview
The folder contains the following:
* data folder which holds relevant data set
* models folder which holds the following: 
  * dictVectorizer
  * trained models: logistic_regression.bin, random_forest_classifier.bin, xgboost_classifier.bin
* notebook.ipynb file containing:
  * exploratory data analysis, which entails a first look at the data, checking of missing values, feature importance analysis, checking for correlation between features. checking imbalance of target variable
  * prototyping of models
  * hyperparameter tuning
* Python scripts
  * train.py, running this script will generate the models. In general, one will not need to use this script except to retrain models with new hyperparameters
  * predict.py, this script deploys a Flask app which accepts post requests with json payload and generates predictions of whether person described in json would have diabetes
  * predict_test.py, this script is to facilitate testing of predict.py. Json posted to flask app can be edited within.
* Dockerfile
* requirements.txt

## How to execute scripts (with Docker)
To start with, you will need to either fork this repository or copy the following folders and files into your directory: **models folder, Dockerfile, requirements.txt, predict.py, predict_test.py**,

To build the image based off dockerfile, within the folder in which dockerfile is held, run the following:

```bash
docker build -t [name of image] .
```

Follow up with this to run a container based on the image with the port exposed:
```bash
docker run -d --rm -p 9696:9696 [name of image]
```

One can then send a request and get the probability of diabetes from the flask app by running the relevant python script via **another** command line interface after changing to a directory which holds the predict_test.py script
```python
python -m predict_test
```

## How to execute scripts (without Dockers)

### Predicting diabetes with pre-trained models
To start with, you will need to either fork this repository or copy the following folders and files into your directory: **models folder, requirements.txt, predict.py, predict_test.py**,

Within the folder in which the files are held, run the following to ensure the requirements are fulfilled:

```bash
pip install -r requirements.txt
```

Get the flask app up and running:
```python
python -m predict
```

One can then send a request and get the probability of diabetes from the flask app by running the relevant python script via **another** command line interface after changing to a directory which holds the predict_test.py script
```python
python -m predict_test
```

### Re-training models
To retrain the model(s), you would require the **data folder**, along with **train.py** script. You could tinker with the parameters within train.py and then run the following in your command line in the folder in which both are stored:
```python
python -m train
```

## Further Discussion
