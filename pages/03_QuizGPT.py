# Imports ──────────────────────────────────────────────────────────────
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

# Page-wide Streamlit settings
st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

# Build (and cache) a retriever from the uploaded file
@st.cache_resource(show_spinner="Loading file…")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

with st.sidebar:
    choice = st.selectbox("Choose what you want to use.", (
        "File","Wikipedia Article"),)

    if choice == "File":
        file = st.file_uploader("Upload a .docx, .txt, or .pdf file", type=["docx", "txt", "pdf"])
        if file:
            docs = split_file(file)
            st.write(docs)
        
    else:
        topic = st.text_input("Search Wikipedia for a topic", placeholder="Enter a topic to search")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)
                st.write(docs)
        

st.markdown(
    """
    This app allows you to upload a document and generate quiz questions based on its content.
    The questions can be used for educational purposes, such as testing knowledge or preparing for exams.
    """
)