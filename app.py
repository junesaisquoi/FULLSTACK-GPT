import os
import pathlib
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Streamlit page config
st.set_page_config(page_title="DocumentGPT", page_icon="📄")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
# Callback that streams tokens to the UI
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    def on_llm_start(self, *_, **__):
        self.message_box = st.empty()
    def on_llm_new_token(self, token, *_, **__):
        self.message += token
        self.message_box.markdown(self.message)
    def on_llm_end(self, *_, **__):
        save_message(self.message, "ai")
        
# Sidebar widgets
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    file    = st.file_uploader("Upload .txt / .pdf / .docx", type=["txt", "pdf", "docx"])
    st.markdown("---")
    st.markdown("[GitHub repo](https://github.com/junesaisquoi/FULLSTACK-GPT)")

# Stop until a key is provided
if not api_key:
    st.info("Enter your OpenAI API key in the sidebar.")
    st.stop()
    
# Language model instance
llm = ChatOpenAI(
    openai_api_key=api_key,
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

# Create or load a retriever from the uploaded file
@st.cache_resource(show_spinner="Embedding file…")
def embed_file(uploaded_file, key):
    # 1) Make sure the local cache directory exists
    files_dir = pathlib.Path("./.cache/files")
    files_dir.mkdir(parents=True, exist_ok=True)

    # 2) Write the uploaded bytes to disk
    file_path = files_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # 3) Choose a loader based on extension
    suffix = uploaded_file.name.lower().split(".")[-1]
    if suffix == "txt":
        loader = TextLoader(str(file_path))
    elif suffix == "pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == "docx":
        loader = Docx2txtLoader(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

    # 4) Load raw documents
    raw_docs = loader.load()

    # 5) Split into chunks with CharacterTextSplitter
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    docs = text_splitter.split_documents(raw_docs)

    # 6) Create / load embeddings cache
    store_dir = pathlib.Path(f"./.cache/embeddings/{uploaded_file.name}")
    store_dir.parent.mkdir(parents=True, exist_ok=True)
    embeddings = CacheBackedEmbeddings.from_byte_store(
        OpenAIEmbeddings(openai_api_key=key),
        LocalFileStore(str(store_dir)),
    )

    # 7) Build a FAISS index and return a retriever
    return FAISS.from_documents(docs, embeddings).as_retriever()

# Session-state helpers
def save_message(msg, role):
    st.session_state["messages"].append({"message": msg, "role": role})

def send_message(msg, role, save=True):
    with st.chat_message(role):
        st.markdown(msg)
    if save:
        save_message(msg, role)

def paint_history():
    for m in st.session_state["messages"]:
        send_message(m["message"], m["role"], save=False)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# RAG prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using ONLY the following context. "
     "If you don't know the answer, say you don't know. "
     "Do not make anything up.\n\nContext:\n{context}"),
    ("human", "{question}")
])

# Main UI
st.title("📄 DocumentGPT")
st.markdown("Upload a document and ask questions about its content.")

if file:
    retriever = embed_file(file, api_key)
    send_message("File processed! Ask me anything.", "ai", save=False)
    paint_history()

    user_q = st.chat_input("Type your question here…")
    if user_q:
        send_message(user_q, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(user_q)
else:
    st.session_state["messages"] = []