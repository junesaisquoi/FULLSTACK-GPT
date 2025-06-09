import asyncio
from langchain_community.document_loaders import AsyncChromiumLoader
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from langchain.document_transformers import Html2TextTransformer
import streamlit as st

# Page-wide Streamlit settings
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐"
)
st.title("SiteGPT")

html2text_transformer = Html2TextTransformer()
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
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(docs)