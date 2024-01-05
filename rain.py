import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and dataframe
rain_pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df_rain.pkl', 'rb'))

st.title("Tomorrow Rain Prediction")

a = st.selectbox('Location', df['Location'].unique())
b = st.selectbox('mintemp', df['MinTemp'].unique())
c = st.selectbox('maxtemp', df['MaxTemp'].unique())
d = st.selectbox('rainFall', df['Rainfall'].unique())
e = st.selectbox('windGustDir', df['WindGustDir'].unique())
f = st.selectbox('windGustSpeed', df['WindGustSpeed'].unique())
g = st.selectbox('windDir9am', df['WindDir9am'].unique())
h = st.selectbox('windDir3pm', df['WindDir3pm'].unique())
i = st.selectbox('windSpeed9am', df['WindSpeed9am'].unique())
j = st.selectbox('windSpeed3pm', df['WindSpeed3pm'].unique())
k = st.selectbox('humidity9am', df['Humidity9am'].unique())
l = st.selectbox('humidity3pm', df['Humidity3pm'].unique())
m = st.selectbox('pressure9am', df['Pressure9am'].unique())
n = st.selectbox('pressure3pm', df['Pressure3pm'].unique())
o = st.selectbox('temp9am', df['Temp9am'].unique())
p = st.selectbox('temp3pm', df['Temp3pm'].unique())
q = st.selectbox('rainToday', df['RainToday'].unique())

# Create a DataFrame with the user input
input_data = pd.DataFrame({
    'Location': [a],
    'MinTemp': [b],
    'MaxTemp': [c],
    'Rainfall': [d],
    'WindGustDir': [e],
    'WindGustSpeed': [f],
    'WindDir9am': [g],
    'WindDir3pm': [h],
    'WindSpeed9am': [i],
    'WindSpeed3pm': [j],
    'Humidity9am': [k],
    'Humidity3pm': [l],
    'Pressure9am': [m],
    'Pressure3pm': [n],
    'Temp9am': [o],
    'Temp3pm': [p],
    'RainToday': [q]
})

if st.button('Predict'):
    # Make prediction using the loaded model
    prediction = rain_pipe.predict(input_data)
    probabilities = rain_pipe.predict_proba(input_data)

    # Display the prediction and probabilities
    prediction_label = "Yes" if prediction[0] == 1 else "No"
    st.title(f"Prediction: {prediction_label}")
    
    # Convert keys to strings for JSON serialization
    prob_dict = {str(class_label): prob for class_label, prob in zip(rain_pipe.classes_, probabilities[0])}
    
    st.write("Probability:")
    st.write(prob_dict)