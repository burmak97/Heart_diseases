import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


#two tipes of num pipelines with standart diviation and MinMax normalization
num_pipeline_std = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    #('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])
num_pipeline_mmx = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    #('attribs_adder', CombinedAttributesAdder()),
    ('mmx_scaler', MinMaxScaler())
])

def pipeline(dataset, num_attributes, cat_attributes, std = True):
    if std:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline_std, num_attributes),
            ("cat", OneHotEncoder(), cat_attributes),
            ])
    else:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline_mmx, num_attributes),
            ("cat", OneHotEncoder(), cat_attributes),
            ])
    return full_pipeline.fit_transform(dataset)

def binary(a,b):
    for i in b:
        a[i] =  a[i].map({"Yes" : 1, "No" : 0})
    return(a)
def sex(a,b):
    a[b] =  a[b].map({"Male" : 1, "Female" : 0})
    return(a)
def age(a,b):
    a[b] =  a[b].map({'55-59' : 57, '80 or older': 85, '65-69' : 67, '75-79' : 77, '40-44' : 42, '70-74' : 72, '60-64' : 62, '50-54' : 52, '45-49' : 47, '18-24' : 21, '35-39' : 37, '30-34' : 32, '25-29' : 27})
    return(a)