from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ").replace("CloseSearch Submit Blog", "")

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
)
    loader = SitemapLoader(url,
                           filter_urls=[
                               r"^(.*\/blog\/).*"
                               ],
                           )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    st.write(docs)
    return docs

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
        docs = load_website(url)