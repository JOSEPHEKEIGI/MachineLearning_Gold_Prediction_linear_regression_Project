
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from prophet import Prophet

def GetDataAndClean(FilePath):
    data = FilePath
    DataView = pd.read_csv(data)
    data = DataView.drop('Unnamed: 0',axis=1)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Price_Spread'] = data['High'] - data['Low']
    return data


def Run_Model(data):
    Closed_model = Prophet()
    data.rename(columns={'Date':'ds','Close':'y'},inplace=True)
    Closed_model.fit(data)  
    X = data[['Volume', 'Open', 'High', 'Low','Price_Spread']]
    y = data['y']
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model = model.fit(X_train,y_train)
    return model


data = GetDataAndClean(r'C:\Users\Cecilia Mwaura\Documents\Gold_Prediction_Project\Assets\goldstock.csv')
Run_Model(data)

    


