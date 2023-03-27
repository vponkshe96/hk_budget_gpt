import tiktoken
from langchain.vectorstores import Pinecone
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def fetch_embeddings(index_name, embeddings):
    db = Pinecone.from_existing_index(index_name = index_name, embedding= embeddings)
    return db


def chain_config(db, chain_type, OPENAI_API_KEY):
    llm = OpenAI(openai_api_key= OPENAI_API_KEY, temperature= 0,streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])) 
    qa = RetrievalQA.from_chain_type(llm = llm, chain_type = chain_type, retriever=db.as_retriever())
    return qa
