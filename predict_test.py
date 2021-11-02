#!/usr/bin/env python
# coding: utf-8

import requests
import pprint as pp

url = 'http://localhost:9696/predict'
test = {'age': 47,
        'gender': "Male",
        'polyuria': "No",
        'polydipsia': "No",
        'sudden weight loss': "No",
        'weakness': "No",
        'polyphagia': "No",
        'genital thrush': "No",
        'visual blurring': "No",
        'itching': "Yes", 
        'irritability': "No",
        'delayed healing': "No",
        'partial paresis': "No",
        'muscle stiffness': "No",
        'alopecia': "Yes",
        'obesity': "No"
        }

response = requests.post(url, json=test).json()
pp.pprint(response)