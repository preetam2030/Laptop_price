# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 01:55:05 2024

@author: preet

#Dataset link:
https://www.kaggle.com/datasets/muhammetvarl/laptop-price
"""

# importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

#Reading the data from the csv file
df = pd.read_csv('laptop_price.csv',encoding='latin-1')
print(df.head())
#Checking for empty values
print(df.info())
#No empty values found so progressing without imputation

#dropping the laptop id
df.drop('laptop_ID',axis=1,inplace=True)

#removing GB and kg from the Ram and Weight columns and changing its datatype
df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

# Using One Hot Encoder for converting categorical values into numerical values
categorical_features = ["Company", "Product", "TypeName", "ScreenResolution", "Cpu", "Memory", "Gpu", "OpSys"]
encoder = OneHotEncoder(sparse_output=False)
categorical_encoded = encoder.fit_transform(df[categorical_features])

#Adding the encoded columns to the rest of the features
numerical_features = ["Inches", "Ram", "Weight"]
X = np.hstack((df[numerical_features].values, categorical_encoded))
# creating the label column
y = df["Price_euros"].values

#Splitting the dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Scaling the training and testing dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Create Linear Regression object
model = LinearRegression()

#fitting the model on training data
model.fit(X_train, y_train)

#finding the score on the training data
print(model.score(X_train, y_train)) 

#predicting the price for testing data
y_pred = model.predict(X_test)  

#recording the mean squared error  
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#Dump the trained model in a Pickle file along with scaler, encoder, numerical and categorical column names
with open("models/model.pkl", "wb") as model_file:
    pickle.dump((model, scaler, encoder, numerical_features, categorical_features), model_file)

#creating a test data to predict
test_data = pd.DataFrame({
    "Company": ["Apple"],
    "Product": ["MacBook Pro"],
    "TypeName": ["Ultrabook"],
    "Inches": [13.3],
    "ScreenResolution": ["IPS Panel Retina Display 2560x1600"],
    "Cpu": ["Intel Core i5 3.1GHz"],
    "Ram": ["8GB"],
    "Memory": ["256GB SSD"],
    "Gpu": ["Intel Iris Plus Graphics 650"],
    "OpSys": ["macOS"],
    "Weight": ["1.37kg"],
})
#Load the trained model. (Pickle file)
with open("models/model.pkl", "rb") as model_file:
    saved_model, saved_scaler, saved_encoder, saved_numerical_features, saved_categorical_features = pickle.load(model_file)

#Preprocessing done to the data
test_data["Ram"] = test_data["Ram"].str.replace("GB", "").astype(int)
test_data["Weight"] = test_data["Weight"].str.replace("kg", "").astype(float)
test_categorical_encoded = saved_encoder.transform(test_data[saved_categorical_features])
test_numerical = test_data[numerical_features].values
test_X = np.hstack((test_numerical, test_categorical_encoded))
test_X_scaled = saved_scaler.transform(test_X)

#price is predicted
print(f"Price for given inputs: {saved_model.predict(test_X_scaled)}")
