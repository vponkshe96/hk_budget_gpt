from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import tiktoken_len
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import streamlit as st


loader = PyPDFLoader('budget2023.pdf')
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

chunks = text_splitter.split_documents(docs)

print(f'Doc splitting successfully completed! {len(docs)} page doc has been split into {len(chunks)} chunks')


embeddings = OpenAIEmbeddings(openai_api_key=st.secrets.OPENAI_API_KEY)
pinecone.init(
  api_key=st.secrets.PINECONE_API_KEY, 
  environment=st.secrets.PINECONE_API_ENV  
)
index_name = 'doc-embeddings'
db = Pinecone.from_documents(documents = chunks, embedding = embeddings, index_name = index_name)
print(f"Data ingestion successfully completed! Chunks have been embedded onto Pinecone, DB reference is {db}")