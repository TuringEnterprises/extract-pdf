from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import streamlit as st
import io
import tempfile

def read_pdf(file):
    pdf = PdfFileReader(file)
    text = ""
    for page in range(pdf.getNumPages()):
        text += pdf.getPage(page).extractText()
    return text

examiner_prompt = """
You are an amazing data scientist good at cleaning up medical data.
"""

user_prompt = """
  Raw claim text:
  ```
  {raw_claim}
  ```
  - Reconstruct the claims from the raw text from a PDF file.
  - There are empty fields, and you should copy or infer such fields from the previous claim.
  - If a number is surounded by (), it's a negative number. ($1000) = -$1000
  - Generate the output data as a CSV file, where 1st row is for header information, and rest rows are claim data:
"""

user_prompt_with_table_info = """
  Raw claim text:
  ```
  {raw_claim}
  ```
  - Reconstruct the claims from the raw text from a PDF file.
  - There are empty fields, and you should copy or infer such fields from the previous claim.
  - If a number is surounded by (), it's a negative number. ($1000) = -$1000
  - Additional table information: {table_info}
  - Generate the output data as a CSV file, where 1st row is for header information, and rest rows are claim data:

"""
# - Table has following columns: Subscriber ID, Rel Code, Major Disease Category, ICD, Paid Date, Incurred Date, Amount Paid

class StreamlitStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    def on_message(self, message):
        super().on_message(message)
        if message["role"] == "assistant":
            st.write(message["content"])

# agent.py
class Chatbot:
    def __init__(self, model_name="gpt-3.5-turbo", table_info=None):
        self.chat = ChatOpenAI(streaming=False, model_name=model_name, temperature=0)
        self.table_info = table_info
        system_message_prompt = SystemMessagePromptTemplate.from_template(examiner_prompt)
        if self.table_info:
            human_message_prompt = HumanMessagePromptTemplate.from_template(user_prompt_with_table_info)
        else:
            human_message_prompt = HumanMessagePromptTemplate.from_template(user_prompt)
        self.chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


    def extract(self, pdf_path):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(pdf_path.read())
            tmp_pdf_path = tmp.name

        loader = UnstructuredPDFLoader(tmp_pdf_path)
        pages = loader.load_and_split()
        # as it's a demo, we just get the 1st page.
        raw_text = pages[0].page_content
        st.write("Completed reading pdf.")
        
        if self.table_info:
            res = self.chat(self.chat_prompt.format_prompt(raw_claim=raw_text, table_info=self.table_info).to_messages())
        else:
            res = self.chat(self.chat_prompt.format_prompt(raw_claim=raw_text).to_messages())
        # st.write(res.content)
        st.write("Completed extracting pdf")
        print(res.content)
        return res.content