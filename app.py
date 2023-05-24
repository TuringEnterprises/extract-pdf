import streamlit as st
import os
import json
import base64
from json.decoder import JSONDecodeError
from agent_bot import Chatbot

def download_button(pdf_message):
    json_str_bytes = pdf_message.encode('utf-8')
    b64 = base64.b64encode(json_str_bytes).decode()
    href = f'<a href="data:text/plain;charset=utf-8;base64,{b64}" download="data.txt">Download JSON as TXT file</a>'
    st.markdown(href, unsafe_allow_html=True)


# Auth
st.sidebar.image("Img/robot_reading_resume.png")
st.sidebar.write("By: [Turing](mailto:{})".format("kai.du@turing.com"))

# App 
st.header("PDF Extractor")
st.info("Hey there! I'm an assistant from Turing for extracting data from PDFs.")

# Model selection
model_mapping = {
    "fast": "gpt-3.5-turbo",
    "high quality": "gpt-4"
}

model_options = list(model_mapping.keys())
selected_option = st.selectbox("Select a model:", model_options)
selected_model = model_mapping[selected_option]

# Table info input
table_info = st.text_input("Enter additional table information (optional):")

# Initialize the Chatbot
chat_bot = Chatbot(model_name=selected_model, table_info=table_info)

uploaded_pdf = st.file_uploader("Upload a PDF file:", type=['pdf'], accept_multiple_files=False)

if uploaded_pdf and os.environ.get("OPENAI_API_KEY"):
    pdf_message = chat_bot.extract(uploaded_pdf)
    # Add the download button to the Streamlit app
    # download_button(pdf_message)
    st.write(pdf_message)
else:
    st.info("Please upload a PDF file, and check your configs.")