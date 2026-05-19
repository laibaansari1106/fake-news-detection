import pickle
import streamlit as st
import numpy as np
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'news_predictor_model.sav')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.sav')

# model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))
st.title("Fake news Prediction System")

# Inputs

title = st.text_input("title")
text=st.text_input("text")

# Prediction
if st.button("Predict"):

    content =  title +" " + text 

    input_data = vectorizer.transform([content])

    prediction = model.predict(input_data)

    if prediction[0] == 'real':
        st.success("Real News")
    elif prediction[0]=='fake':
        st.error("Fake News")
    else:
        st.error("Unexpected prediction output")
