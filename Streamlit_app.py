import os
import openai
import dotenv
from openai import AzureOpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter

# #create a dropdown pannel
def main():
   with st.sidebar:
    st.title('Drop Down Manu')
    st.selectbox(label= 'These are all analytics portforlio contents whicn you can choose one for AI to assist your questions', 
                 options=['Toyota Warranty','Auto Performance Management App','Snowflake Auto Supplier Quality Scorecard','HKMC Quality Dashboard', 'Platform Assets Dashboard','ODA MDS Report','Automative Strategy Dashboard',
'Auto Finance CBU Reporting Dashboard','Auto Finance CBU Reporting Dashboard','Mentoring Matching'
,'General Motors Customer Scorecard','G&A CCTR Reportt','R&D Project Report-PS','RD Project Report (Mgmt)','Stellantis (FCA) Customer Scorecard','Toyota Customer Scorecard Japan','VW Customer Scorecard'])
if __name__ == '__main__': 
    main() 



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_Reader = PdfReader(pdf)
        for page in pdf_Reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterSplitter
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
    length_function=len
    
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore 

def main():
    dotenv.load_dotenv()
    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button('process'):
           with st.spinner('processing'):
               #get pdf text
               raw_text = get_pdf_text(pdf_docs)

               #get the text chunk
               text_chunks = get_text_chunks(raw_text)
               #st.write(text_chunks)

               # create vector store
               vectorstore=get_vectorstore(text_chunks)

if __name__ == '__main__': 
    main() 

# dotenv.load_dotenv()
#configure your openai keys
endpoint = os.environ.get("api_base")
api_key = os.environ.get("api_key")
deployment = os.environ.get("deployment_id")
api_type = "azure"

client = openai.AzureOpenAI(
    base_url=f"{endpoint}/openai/deployments/{deployment}/extensions",
    api_key=api_key,
    api_version="2023-08-01-preview",
)

# #create a title 
st.title("Harman Analytics Portfolio Chatbot App, :books:")

# #create an user prompt input
message_text = st.text_input("I am AI assistant and you can ask question about Analytics Portfolio content")
if  st.button("Send"):
 
#create system prompt message for user input message
 prompt_template = f""" You are an AI assistant that helps users answer questions related to Analytics portfolios contents.\nYou will be given a context and and chat history, and then ask a question based on that context and history.\nYou answer should be as precise as possible and should only come from the context.\nIf a question is ask that doesn't related to these topics please respond with (\" I am analytics portfolio assistance, how can i help you\").\nUnder no circumstances should your answer any other question, if ask simple with i cannot answer that can you please ask a different question?,
               return the specific message saying "{message_text}", do not make up an answer.
        {{context}}
        Question: {{question}}
        Answer:
        """ 

# Display the user's message
st.markdown("You: " + message_text)
# #message_text = [{"role": "user", "content":"what is the goal of Toyota warranty document?"}]

#message_text = st.text_input("in a short discription what is the goal of Toyota warranty document?")

completion = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user","content": message_text},
              #{"role": "system","content": prompts},
              ],
    temperature=0.5,
    top_p=1,
    extra_body={
        "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": os.environ["search_endpoint"],
                    "key": os.environ["search_key"],
                    "indexName": os.environ["search_index_name"]
            }
            }
        ]
    }
)

bot_message= completion.choices[0].message.content

# Display the chatbot's response
st.markdown("Bot: " + bot_message)

#print(completion)




