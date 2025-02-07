import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

chatbot = pipeline("text-generation", model="distilgpt2")

def heathcare_chatbot(user_input):
    if "symptoms" in user_input:
        response="Please provide more details about your symptoms."
    elif "appointment" in user_input:
        response="Please provide your name and contact details."
    elif "prescription" in user_input:
        response="Please provide the name of the medication."
    elif "diagnosis" in user_input:
        response="Please provide more details about your diagnosis."

    else:
        response=chatbot(user_input,max_length=500,num_return_sequences=1)
        return response[0]['generated_text']    


def main():
    st.title("Healthcare Assistant Chatbot") 
    user_input=st.text_input("How can I assist you today?")
    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            response=heathcare_chatbot(user_input)
            st.write("Heathcare Assistant: ", response)
    else:
        st.write("Please enter a message to get a response.")    
main()