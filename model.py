"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction...  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
#########################################################
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import AdaBoostRegressor

# Model slection
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# Other
from scipy import stats
import math
import pickle
#########################################################

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    #feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    train_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    train_df = train_df[(train_df['Commodities'] == 'APPLE GOLDEN DELICIOUS')]
    
    #Changing date for train
    train_df['Date']= pd.to_datetime(train_df['Date'])
    train_df['Months'] = train_df['Date'].dt.strftime('%B')
    train_df['Year'] = train_df['Date'].dt.strftime('%Y')
    
    #Date
    train_df['year'] = pd.DatetimeIndex(train_df['Date']).year
    train_df['Months'] = pd.DatetimeIndex(train_df['Date']).month
    
    #Changing date for train
    train_df['Date']= pd.to_datetime(train_df['Date'])
    train_df['Date'] = train_df['Date'].dt.strftime('%B')
    
    #Avarage must be the last column (dependent)
    #reorder columns
    column_titles = [col for col in train_df.columns if col!= 'avg_price_per_kg'] + ['avg_price_per_kg']
    train_df=train_df.reindex(columns=column_titles)
    
    #spliting variables into x and y
    predict_vector = train_df[['Sales_Total','Low_Price','High_Price','Stock_On_Hand','Weight_Kg']]
    Y = train_df['avg_price_per_kg']
    
    
    
    
    #------feature_vector_df = feature_vector_df[(feature_vector_df['Commodities'] == 'APPLE GOLDEN DELICIOUS')]
    #------predict_vector = feature_vector_df[['Total_Qty_Sold','Stock_On_Hand']]
                                
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
