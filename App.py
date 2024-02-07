import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def GetData(filePath):
    data = pd.read_csv(filePath)
    data['Price_Spread'] = data['High'] - data['Low']
    return data


def model(data):
    X = data[['Volume', 'Open', 'High', 'Low', 'Price_Spread']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.coef_, model.intercept_


def Model_equation(attrib, value):
    coef, intercept = attrib
    return coef[value], intercept


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
    st.write(f'{selected_feature} = {coef_value} * Close + {intercept_value}')

   
    st.subheader('Predict Close Price')
    volume = st.number_input('Volume')
    open_price = st.number_input('Open Price')
    high_price = st.number_input('High Price')
    low_price = st.number_input('Low Price')
    price_spread = high_price - low_price
    predicted_close = coef_value * volume + coef_value * open_price + coef_value * high_price + coef_value * low_price + coef_value * price_spread + intercept_value
    st.write(f'Predicted Close Price: {predicted_close}')
