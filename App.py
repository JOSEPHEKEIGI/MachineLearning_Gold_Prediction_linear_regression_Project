import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pyscript import app


def GetData(filePath):
    data = filePath
    DataView = pd.read_csv(data)
    data = DataView.drop('Unnamed: 0',axis=1)
    data['Price_Spread'] = data['High'] - data['Low']
    return data

def model(data):
    X = data[['Volume', 'Open', 'High', 'Low','Price_Spread']]
    y = data['Close']
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model.coef_ , model.intercept_


data = GetData(r'C:\Users\Cecilia Mwaura\Documents\Gold_Prediction_Project\Assets\goldstock.csv')
#print(model(data))
attrib = model(data)

def Model_equation(attrib, value):
    # Y = MX + C
    # Volume = Close * Coef + Intercept
    # ['Volume', 'Open', 'High', 'Low','Price_Spread']
    coef, intercept = attrib
    return coef[value] , intercept

features={
    'Volume':0,
    'Open':1,
    'High':2,
    'Low':3,
    'Price_Spread':4
}
 

print(Model_equation(attrib,features['Volume']))



    
    
    
    

 


    

    









