import streamlit as st
import pickle

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("Emotion Detection from Text")

user_input = st.text_area("Enter your text")

if st.button("Predict Emotion"):
    data = vectorizer.transform([user_input])
    prediction = model.predict(data)
    st.success("Emotion: " + prediction[0])
