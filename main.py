from utils import fetch_embeddings,chain_config
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import streamlit as st
from test import test_answer


#initializing embeddings and pinecone class
pinecone.init(
    api_key=st.secrets.PINECONE_API_KEY,
    environment=st.secrets.PINECONE_API_ENV
)
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets.OPENAI_API_KEY)
index_name = "doc-embeddings"
db = fetch_embeddings(index_name= index_name, embeddings= embeddings)

#setting up chain
qa_chain = chain_config(db=db, chain_type= "stuff", OPENAI_API_KEY=st.secrets.OPENAI_API_KEY)

#Frontend
# Importing and reading content of css file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Create an HTML tag to link style.css file
html_string = f'<link rel="stylesheet" type="text/css" href="./style.css"/>'
st.markdown(html_string, unsafe_allow_html=True)
#Importing icons library
icons_css = f'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">'
st.markdown(icons_css, unsafe_allow_html=True)
# Define the HTML code to embed the visitor counter
html_code = """
<!-- Start of StatCounter Code -->
<script type="text/javascript" src="//c.statcounter.com/12862567/0/9054f111/0/"></script>
<noscript><div class="statcounter"><a title="Web Analytics Made Easy - StatCounter"
href="http://statcounter.com/" target="_blank"><img class="statcounter"
src="//c.statcounter.com/12862567/0/9054f111/0/" alt="Web Analytics Made Easy -
StatCounter"></a></div></noscript>
<!-- End of StatCounter Code -->
"""
# getting rid of whitespace on top
st.write('<style>div.block-container{padding-top:0px;}</style>', unsafe_allow_html=True)

st.write('<div class = "heading">🇭🇰 I am HK Budget 2023 ChatGPT 🤖 </div>',unsafe_allow_html=True)

st.write('<div class = "subheading">🤔 Ask me anything </div>',unsafe_allow_html=True)

query = st.text_input(label = " ", placeholder= "What is the gov doing for the tech sector?")

if query:
    with st.spinner('Wait for it...'):
        st.write('<div class = "subheading">💡 Answer </div>',unsafe_allow_html=True)
        # answer = qa_chain.run(query)
        st.markdown(f"<div class='rounded-box'>{test_answer}</div>", unsafe_allow_html=True)
        st.markdown(f"<br/>", unsafe_allow_html=True)
        sources = db.similarity_search(query)
        st.write('<div class = "sourcesHeader">🔎 References</div>',unsafe_allow_html=True)
        for source in sources:
            page = int(source.metadata['page']) + 1
            st.write(f"<div class = 'sources'><a href = 'https://www.budget.gov.hk/2023/eng/pdf/e_budget_speech_2023-24.pdf#page={page}' > Page {page}</a></div>", unsafe_allow_html= True)
        st.text(" ")
        st.write('<div class = "footer"><span class = "footerContent">Created by Vinit Ponkshe (SWE)</span><a href ="https://www.linkedin.com/in/vinitponkshe/"><i class="fa-brands fa-linkedin icon " style="color: #0f62f0;"></i></a><span class = "footerContent">& Grace Tang (PM) </span><a href ="https://www.linkedin.com/in/tantingtang/"><i class="fa-brands fa-linkedin icon" style="color: #0f62f0;"></i></a></div> ',unsafe_allow_html=True)
        # st.markdown(f'{html_code}',unsafe_allow_html= True)
        


    
