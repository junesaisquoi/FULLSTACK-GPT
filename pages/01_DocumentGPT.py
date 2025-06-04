import time
import streamlit as st

st.title("DocumentGPT")

with st.chat_message("human"):
    st.write("Hello")
    
with st.chat_message("ai"):
    st.write("Hi, how are you?")
    
with st.status("Embedding file...", expanded=True) as status:
    time.sleep(2)
    st.write("Fetching the file")
    time.sleep(2)
    st.write("Embedding the file")
    time.sleep(2)
    st.write("Caching the file")
    status.update(label="Error", state="error")

st.chat_input("Send a message to AI")