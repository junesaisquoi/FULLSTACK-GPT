import streamlit as st

st.title("PrivateGPT")# Imports ──────────────────────────────────────────────────────────────
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

# Page-wide Streamlit settings
st.set_page_config(
    page_title="PrivateGPT",
    page_icon="🔐"
)

# Ensure chat history exists
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
# Streaming callback to show tokens live
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()  
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# Create OpenAI chat model with streaming + callback
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler()
    ]
)

# Build (and cache) a retriever from the uploaded file
@st.cache_resource(show_spinner="Embedding file…")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# Helper: add message to session state
def save_message(message,role):
    st.session_state["messages"].append({"message":message, "role":role})

# Helper: render a message and optionally store it
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        
# Helper: repaint entire chat history
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False
        )

# Helper: join retrieved docs together
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
        
# Prompt template with context placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    Answer the question using ONLY the following context. If you don't know the answer, just say you don't know. DON'T make anything up.
    
    Context: {context}
    """
    ),
    ("human","{question}")
])

# Page header and instructions
st.title("DocumentGPT")

st.markdown(
    """
    Welcome!
    Use this chatbot to ask questions to an AI about your files!
    Upload a file on the side bar.
    """
)

# Sidebar: file uploader
with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["txt","pdf","docx"])

# Main chat logic
if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask anything :)", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm
        with st.chat_message("ai"):
            chain.invoke(message)

else:                                               # no file yet → reset chat
    st.session_state["messages"] = []