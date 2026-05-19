import pickle
import streamlit as st
import numpy as np

# Load model
model = pickle.load(open('news_predictor_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
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