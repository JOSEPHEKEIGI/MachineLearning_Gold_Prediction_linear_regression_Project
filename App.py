import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st



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
 

st.title('Linear Regression Model')


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = GetData(uploaded_file)

    st.subheader('Data Preview')
    st.write(data.head())

    st.subheader('Training the Model')
    coefficients, intercept = model(data)
    st.write('Coefficients:', coefficients)
    st.write('Intercept:', intercept)

    selected_feature = st.selectbox('Select a feature:', list(features.keys()))

   
    st.subheader('Model Equation')
    coef_value, intercept_value = Model_equation((coefficients, intercept), features[selected_feature])
    st.write(f'{selected_feature} = {coef_value} * y + {intercept_value}')





    
    
    
    

 


    

    









