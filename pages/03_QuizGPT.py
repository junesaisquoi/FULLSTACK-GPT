# Imports ──────────────────────────────────────────────────────────────
import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "").strip()
        return json.loads(text)

output_parser = JsonOutputParser()

# Page-wide Streamlit settings
st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             """
             You are a helpful assistant that is role-playing as a teacher. 
             Based ONLY on the following context, make 10 questions to test the user's knowledge about the text.
             Each question should have 4 options, three of which must be incorrect and one should be correct.
             Use (o) to signify the correct answer.
             
             Qeustion examples:
             Question 1: What is the color of the ocean?
             Answer: Red|Yellow|Blue (o)|Green
             
             Question 2: What is the capital of Georgia?
             Answer: Baku|Tbilisi (o)|Manila|Beirut
             
             Question 3: When was Avatar released?
             Answer: 2009 (o)|2010|2011|2008
             
             Question: Who was Julius Caesar?
             Answer: A Roman general and statesman (o)|A Greek philosopher|A Chinese emperor|An Egyptian pharaoh
             
             Your turn!
             
             Context: {context}
             """),
        ]
    )

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
     You are a powerful formatting algorithm that formats quiz questions and answers.
     You format exam questions into a JSON format that can be used in a quiz application.
     The options with (o) signify the correct answer.
     Example Input:
      Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
         
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model
        
        Example Output:
        ```json
        {{ "questions": [
                {{
                    "question": "What is the color of the ocean?",
                    "answers": [
                            {{
                                "answer": "Red",
                                "correct": false
                            }},
                            {{
                                "answer": "Yellow",
                                "correct": false
                            }},
                            {{
                                "answer": "Green",
                                "correct": false
                            }},
                            {{
                                "answer": "Blue",
                                "correct": true
                            }},
                    ]
                }},
                            {{
                    "question": "What is the capital or Georgia?",
                    "answers": [
                            {{
                                "answer": "Baku",
                                "correct": false
                            }},
                            {{
                                "answer": "Tbilisi",
                                "correct": true
                            }},
                            {{
                                "answer": "Manila",
                                "correct": false
                            }},
                            {{
                                "answer": "Beirut",
                                "correct": false
                            }},
                    ]
                }},
                            {{
                    "question": "When was Avatar released?",
                    "answers": [
                            {{
                                "answer": "2007",
                                "correct": false
                            }},
                            {{
                                "answer": "2001",
                                "correct": false
                            }},
                            {{
                                "answer": "2009",
                                "correct": true
                            }},
                            {{
                                "answer": "1998",
                                "correct": false
                            }},
                    ]
                }},
                {{
                    "question": "Who was Julius Caesar?",
                    "answers": [
                            {{
                                "answer": "A Roman Emperor",
                                "correct": true
                            }},
                            {{
                                "answer": "Painter",
                                "correct": false
                            }},
                            {{
                                "answer": "Actor",
                                "correct": false
                            }},
                            {{
                                "answer": "Model",
                                "correct": false
                            }},
                    ]
                }}
            ]
        }}
    ```
    Your turn!

    Questions: {context}

     """),
])

formatting_chain = formatting_prompt | llm

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
    docs = None
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
        

if not docs: 
    st.markdown(
        """
        Welcome to QuizGPT!
        This application allows you to generate quiz questions based on the content of a document or a Wikipedia article.
        
        Get started by uploading a document or searching for a topic on Wikipedia in the sidebar.
        Once you have selected a source, you can generate quiz questions based on the content.
        """
    )
else:
    start = st.button("Generate Quiz Questions")
    
    if start:
        chain = {"context":questions_chain} | formatting_chain | output_parser
        response = chain.invoke(docs)
        st.write(response)