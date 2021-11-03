# diabetes_prediction
This project is done as a midterm project for [Alexey Grigorev's mlbookcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project)

### Table of Contents
* [Problem Description](#problem_desc)
* [Repo Structure overview](#repo_overview)
* [Testing Locally](#test_docker)
  * [Executing scripts (with Dockers)](#test_docker)
  * [Executing scripts (with virtual environment)](#test_nodocker)
* [Cloud Deployment](#cloud)
  * [Testing the public endpoint](#cloud_test)
  * [Code for cloud deployment](#cloud_deploy)
* [Further Discussion](#further_disc)
* [References](#references)

<a id='problem_desc'></a>
## Problem Description
Diabetes is a chronic disease which could lead to death and cause other issues such as limb amputation. In fact, World Health Organisation estimates that diabetes has direct caused the deaths of 1.5 million people in 2019. [1] It is then no surprise that one would wish to detect diabetes earlier in order to hope to alleviate the illness and other problems that comes along with it.

In this project, the dataset is gotten from https://www.kaggle.com/ishandutta/early-stage-diabetes-risk-prediction-dataset. According to the author, the data was collected via questionnaires from patients of Sylhet Diabetic Hospital located in Sylhet, Bangladesh. The dataset has 16 features (Age, Gender, other conditions such as polyuria, sudden weight loss etc), along with 1 target variable (class), where positive indicates patient has diabetes.

Using the dataset, three classification models were trained: Logistic Regression, Random Forest Classifier and XGBoost Classifier. Using these pretrained models, one could then use it to predict if patient might have diabetes. Prevention is no doubt better than cure but for those who have it detected early, they could do the necessary lifestyle changes and potentially reverse diabetes. [3]

<a id='repo_overview'></a>
## Repository Structure Overview
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
  * predict_test.py, this script is to facilitate testing of Flask deployed model on your local machine, with or without Docker. Json posted to flask app can be edited within.
  * predict_test_cloud.py, this script is to facilitate testing of Flask deployed model on the cloud (https://fwahh-diabetes-prediction.herokuapp.com/predict). Json posted to flask app can be edited within.
* Dockerfile
* requirements.txt

<a id='test_docker'></a>
## How to execute scripts (with Docker)
To start with, you will need to either fork this repository or copy the following folders and files into your directory: **models folder, Dockerfile, requirements.txt, predict.py, predict_test.py**.

**Please ensure your Docker desktop is up and running before attempting the below**

To build the image based off dockerfile, within the folder in which dockerfile is held, run the following:

```bash
docker build -t [name of image] .
```

Follow up with this to run a container based on the image with the port exposed:
```bash
docker run -d --rm -p 9696:9696 [name of image]
```

One can then send a request and get the probability of diabetes from the flask app by running the relevant python script via **another** command line interface after changing to a directory which holds the predict_test.py script
```bash
python -m predict_test
```
To test with different values, you could change the dictionary test. Note that age should be an integer value, gender should be any value (Male/Female) and the rest of the variables are either 'Yes' or 'No'.

### Examples with Images

In the following examples, \[name of image] is diabetes

Building the image:

![image](https://user-images.githubusercontent.com/65491089/139817724-9eab1b51-e7b8-4682-8368-fff3f89897b0.png)

Running the container:

![image](https://user-images.githubusercontent.com/65491089/139817798-914ec8b7-d804-44b1-bf8a-7ebf38de56a0.png)

Note that `-d` allows the container to run in the background. If you would like to run it interactively, replace `-d` with `-it` and you'll see the following. Please note **not** to run this after you have already run the previous command with `-d` flag. that will cause an error since the port is already allocated and container is running in the background.

![image](https://user-images.githubusercontent.com/65491089/139818006-b50bdd5d-b2d5-4b64-a501-bb8afa275d7e.png)

Testing in a separate CLI:

![image](https://user-images.githubusercontent.com/65491089/139818195-854f200a-1cc4-4304-af30-d96f75b5f82c.png)

After you are done testing, to stop the container, run the command `docker ps` to retrieve container ID and then use `docker stop` to stop it.

![image](https://user-images.githubusercontent.com/65491089/139819322-58c75f48-f128-42d9-a24f-1e029fc55b4b.png)

![image](https://user-images.githubusercontent.com/65491089/139818590-732a51bb-033c-401e-bed3-902dcca98679.png)

If you try to run predict_test.py while container is not up, you'll received a ConnectionRefusedError due to the flask app not being up and running.

<a id='test_nodocker'></a>
## How to execute scripts (with virtual environment)

### Predicting diabetes with pre-trained models
If you do not have docker, you may want to do your testing in a virtual environment. To start with, you will need to either fork this repository or copy the following folders and files into your directory: **models folder, requirements.txt, predict.py, predict_test.py**. Next ensure virtualenv is installed by running the following. If it's not installed, you can do a pip install for the module virtualenv.

```bash
virtualenv --version
```

Within the folder in which the files are held, create your virtual environment diabetes (you can rename your virtualenv by replacing diabetes with something else) as follows:

```bash
python -m venv diabetes
```

Once venv is created you can activate with 1 of the following commands, depending on your operating system:

```bash
# if windows:
diabetes\Scripts\activate

# for non-windows:
diabetes/bin/activate
```
If done correctly, you should see the venv's name in parentheses in front of your CLI prompt, similar to the image below. Note that in this image, my virtual env is called diabetes and I'm on Windows.

![image](https://user-images.githubusercontent.com/65491089/140039797-d8ac1bd4-00a5-45c8-b6ab-3d097eadf1cc.png)

Within the virtual env, run the following to ensure the requirements are fulfilled:

```bash
pip install -r requirements.txt
```

Get the flask app up and running:
```bash
python -m predict
```

One can then send a request and get the probability of diabetes from the flask app by running the relevant python script via **another** command line interface after changing to a directory which holds the predict_test.py script
```bash
python -m predict_test
```

Once done with testing, you can simply deactivate the virtual environment by typing `deactivate` in CLI. You will know it has been deactivated if the name of the venv disappears from your command prompt. Directly closing the command line interface is fine too.

### Re-training models
To retrain the model(s), you would require the **data folder**, along with **train.py** script. You could tinker with the parameters within train.py and then run the following in your command line in the folder in which both are stored. It would be best to do this within the virtual environment from above to ensure all dependencies are installed:
```python
python -m train
```

<a id='cloud'></a>
## Cloud Deployment
<a id='cloud_test'></a>
### Testing the public endpoint
Though the app has been deployed on cloud. Note that https://fwahh-diabetes-prediction.herokuapp.com/predict can't be accessed directly as it only accepts post request. To test, you can directly run predict_test_cloud.py (ensure your CLI is in the folder in which this python script is stored)
```bash
python -m predict_test_cloud
```
If ran correctly, you should see something similar to the following:
![image](https://user-images.githubusercontent.com/65491089/139905424-e6d1458a-f9e7-45c3-a998-97408cb48767.png)

To test with different values, you could change the dictionary test. Note that age should be an integer value, gender should be any value (Male/Female) and the rest of the variables are either 'Yes' or 'No'.

<a id='cloud_deploy'></a>
### Code for cloud deployment

Note you do not have to follow the steps detailed below in order to test. They are mentioned here for reference on how to deploy one's app to the cloud, more specifically Heroku.

First, ensure Heroku CLI is downloaded and heroku is added to path. Thereafter you could navigate to the folder in which the necessary folders and files are stored. Once in the folder, start with a heroku login command, press any key except q to open up the browser. The browser can be closed after logged in screen shows.

```bash
heroku login
```

Now you can run the following commands in succession.

```bash
heroku container:login
heroku create [name of app]
heroku container:push web -a [name of app]
heroku container:release web -a [name of app]
```
The above commands logs you in to the Heroku container registry, creates an app with a customized name that will also be the url of your web app, i.e. your app will be hosted on \[name of app]/herokuapp.com. In my case, I called it *fwahh-diabetes-prediction*. The third command pushes the docker image to Hero container registry. The image will be built based on the Dockerfile in the current working directory. The final command releases the image previously built to the web.

Special thanks to Ninad Date's guide on how to deploy apps on heroku. You can check out the guide here: https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-heroku.md

<a id='further_disc'></a>
## Further Discussion

On the test set, the following scores were achieved. 

| Model        | ROC-AUC score   | Accuracy  |
| ------------ | --------------- | --------- |
| Logistic Regression   | 0.979 | 0.923 |
| Random Forest Classifier | 1.0 | 0.990 |
| XGBoostClassifier | 1.0 | 1.0|

Honestly, the models all performed well, way out of my expectations. Between the models, it is of no surprise that XGBoost performed the best out of them all. Before placing too much trust in these models, it would be great to get more data from both patients in Sylhet Diabetic Hospital, and from patients worldwide, just to see if the models performed just as well on them. I've checked that there was no data leakage from the test set when exploring the data, or training the models. However, if you spot any issues, do let me know! :)

<a id='references'></a>
## References
[1] : https://www.who.int/news-room/fact-sheets/detail/diabetes

[2] : https://www.kaggle.com/ishandutta/early-stage-diabetes-risk-prediction-dataset

[3] : https://www.webmd.com/diabetes/can-you-reverse-type-2-diabetes
