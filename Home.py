import streamlit as st

#page title
st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🚀"
)

#title
st.title("FullstackGPT Home")

#sidebar
with st.sidebar:
    st.title("sidebar title")
    st.text_input("xxx")
    
#tabs
tab_one, tab_two, tab_three = st.tabs(["A","B","C"])

with tab_one:
    st.write("a")
    
with tab_two:
    st.write("b")

with tab_three:
    st.write("c")