from langchain_community.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import streamlit as st

llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
    
    Then, give a scorer to the answer between 0 and 5. 0 being not helpful to the user and 5 being helpful to the user.
    
    Make sure to include the answer's score.
    
    Context: {context}

    Examples:
    
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away from Earth.
    Score: 5
    
    Question: How far way is the sun?
    Answer: I don't know.
    Score: 0
    
    Your turn!
    
    Question: {question}
    """
    )
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    answers = []
    for doc in docs:
        result = answers_chain.invoke({
            "question": question,
            "context": doc.page_content
        })
        answers.append(result.content)
    st.write(answers)

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ").replace("CloseSearch Submit Blog", "")

@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
)
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )                   
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore.as_retriever()

# Page-wide Streamlit settings
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐"
)
st.title("SiteGPT")

# Page header and instructions
st.markdown(
    """
    Ask questions about the content of a website using SiteGPT.
    Enter the URL of the website you want to analyze in the sidebar, and then ask your questions below.
    """
)

# Sidebar for URL input
with st.sidebar:
    url = st.text_input("Enter the URL of the website you want to analyze:", placeholder="https://example.com")
    
if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a sitemap URL (e.g., https://example.com/sitemap.xml).")
    else:
        retriever = load_website(url)
        chain = {"docs": retriever, "question": RunnablePassthrough()} | get_answers | RunnableLambda(get_answers)
        
        chain.invoke("What is the pricing of GPT-4 Turbo with vision?")