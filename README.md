# diabetes_prediction
This project is done as a midterm project for [Alexey Grigorev's mlbookcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project)

## Problem Description
Diabetes is a chronic disease which could lead to death and cause other issues such as limb amputation. In fact, World Health Organisation estimates that diabetes has direct caused the deaths of 1.5 million people in 2019. [1] It is then no surprise that one would wish to detect diabetes earlier in order to hope to alleviate the illness and other problems that comes along with it.

In this project, the dataset is gotten from https://www.kaggle.com/ishandutta/early-stage-diabetes-risk-prediction-dataset. According to the author, the data was collected via questionnaires from patients of Sylhet Diabetic Hospital located in Sylhet, Bangladesh. The dataset has 16 features (Age, Gender, other conditions such as polyuria, sudden weight loss etc), along with 1 target variable (class), where positive indicates patient has diabetes.

Using the dataset, three classification models were trained: Logistic Regression, Random Forest Classifier and XGBoost Classifier. Using these pretrained models, one could then use it to predict if patient might have diabetes. Prevention is no doubt better than cure but for those who have it detected early, they could do the necessary lifestyle changes and potentially reverse diabetes. [3]

## Folder structure overview
This repository contains the following:
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
### Examples with Images

In the following examples, \[name of image] is diabetes

Building the image:

![image](https://user-images.githubusercontent.com/65491089/139817724-9eab1b51-e7b8-4682-8368-fff3f89897b0.png)

Running the container

![image](https://user-images.githubusercontent.com/65491089/139817798-914ec8b7-d804-44b1-bf8a-7ebf38de56a0.png)

Note that -d allows the container to run in the background. If you would like to run it interactively, replace -d with -it and you'll see the following. Please note **not** to run this after you have already run the previous command with -d flag. that will cause an error since the port is already allocated and container is running in the background.

![image](https://user-images.githubusercontent.com/65491089/139818006-b50bdd5d-b2d5-4b64-a501-bb8afa275d7e.png)

Testing in a separate CLI

![image](https://user-images.githubusercontent.com/65491089/139818195-854f200a-1cc4-4304-af30-d96f75b5f82c.png)

After you are done testing, to stop the container, run the command "docker ps" to retrieve container ID and then use docker stop to stop it.

![image](https://user-images.githubusercontent.com/65491089/139819322-58c75f48-f128-42d9-a24f-1e029fc55b4b.png)

![image](https://user-images.githubusercontent.com/65491089/139818590-732a51bb-033c-401e-bed3-902dcca98679.png)

If you try to run predict_test.py while container is not up, you'll received a ConnectionRefusedError due to the flask app not being up and running.

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

On the test set, the following scores were achieved. 

| Model        | ROC-AUC score   | Accuracy  |
| ------------ | --------------- | --------- |
| Logistic Regression   | 0.979 | 0.923 |
| Random Forest Classifier | 1.0 | 0.990 |
| XGBoostClassifier | 1.0 | 1.0|

Honestly, the models all performed well, way out of my expectations. Between the models, it is of no surprise that XGBoost performed the best out of them all. Before placing too much trust in these models, it would be great to get more data from both patients in Sylhet Diabetic Hospital, and from patients worldwide, just to see if the models performed just as well on them. I've checked that there was no data leakage from the test set when exploring the data, or training the models. However, if you spot any issues, do let me know! :)

## References
[1] : https://www.who.int/news-room/fact-sheets/detail/diabetes

[2] : https://www.kaggle.com/ishandutta/early-stage-diabetes-risk-prediction-dataset

[3] : https://www.webmd.com/diabetes/can-you-reverse-type-2-diabetes
