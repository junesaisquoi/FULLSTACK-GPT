import streamlit as st
from langchain.prompts import PromptTemplate


st.write("hello")

#1
st.write([1,2,3,4])
st.write({"x":1})

#2
a = [1,2,3,4]
d = {"x": 1}
a
d

#1
st.write(PromptTemplate)

#2
p = PromptTemplate.from_template("xxxx")
st.write(p)


st.selectbox("Choose your model", ("GPT-3", "GPT-4"))
