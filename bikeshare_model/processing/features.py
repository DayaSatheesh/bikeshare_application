from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variable: str, day: str, dteday: str):
        self.variable = variable
        self.nan_indices = []
        self.day = day
        self.dteday = dteday

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        print(X.info())
        X[self.day] = pd.to_datetime(X[self.dteday]).dt.day_name()
        self.nan_indices = X.loc[X.weekday.isnull(), self.variable].index
        return self
          
    def transform(self, data: pd.DataFrame):
        #print(self.nan_indices)
        X = data.copy()   
        X.loc[self.nan_indices, self.variable] = X.loc[self.nan_indices, self.day].apply(lambda x: x[:3])                
        X.drop(self.dteday, axis=1, inplace=True)
        X.drop(self.day, axis=1, inplace=True)
        return X

class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variable:str):
        self.variable = variable

    def fit(self, data: pd.DataFrame, y: pd.Series = None):
        X = data.copy()
        return self

    def transform(self, data: pd.DataFrame):
        X = data.copy()
        mode = X['weathersit'].mode()[0]
        print (mode)
        X[self.variable] = X[self.variable].fillna(mode)        
        return X       


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mapping_values: dict):
        self.variables = variables
        self.mapping_values = mapping_values

    def fit(self, data: pd.DataFrame, y: pd.Series = None):
        #['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
        return self

    def transform(self, data: pd.DataFrame):
        X = data.copy()

        X[self.variables] = X[self.variables].map(self.mapping_values).astype(int)
        return X
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables: str):
        self.variables = variables
        

    def fit(self, data: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = data.copy()
        q1 = X.describe()[self.variables].loc['25%']
        q3 = X.describe()[self.variables].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        return self

    def transform(self, data: pd.DataFrame):
        X = data.copy()

        for i in X.index:
            if X.loc[i, self.variables] > self.upper_bound:
                X.loc[i, self.variables]= self.upper_bound
            if X.loc[i, self.variables] < self.lower_bound:
                X.loc[i, self.variables]= self.lower_bound

        return X
        
        

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variable:str):
        self.variable = variable
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, data: pd.DataFrame, y: pd.Series = None):
        X = data.copy()
        self.encoder.fit(X[[self.variable]])
        # Get encoded feature names
        self.encoded_features_names = self.encoder.get_feature_names_out([self.variable])
        #print(self.encoded_features_names)
        return self

    def transform(self, data:pd.DataFrame):
        X = data.copy()

        encoded_weekdays = self.encoder.transform(X[[self.variable]])
        #print (encoded_weekdays)
        # Append encoded weekday features to X
        X[self.encoded_features_names] = encoded_weekdays

        # drop 'weekday' column after encoding
        X.drop(self.variable, axis=1, inplace=True)

        return X
