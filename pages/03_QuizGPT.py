# ───────────────────────── 1 · Imports ──────────────────────────
import json
import streamlit as st
from pathlib import Path

import os, hashlib

# ── Unicode‑safe OpenAI headers ──
import httpx

def _utf8_header(value, encoding=None):
    if isinstance(value, str):
        return value.encode("utf-8")  # allow any Unicode
    return value

httpx._models._normalize_header_value = _utf8_header  # type: ignore

# LangChain core
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import BaseOutputParser

# Community integrations
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever

# ─────────────────────── 2. Streamlit page config ────────────────────
st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)
st.title("QuizGPT")
# ───────────────────── 3 · Sidebar controls ──────────────────────
with st.sidebar:
    st.markdown("### Data source & settings")
    
    # 3‑a · OpenAI API key (user‑supplied)
    api_key = st.text_input("OpenAI API key", type="password")
    
    # 3‑b · Difficulty selector
    difficulty = st.selectbox("Question difficulty", ("easy", "hard"))
    
    # 3‑c · Source type
    source_type = st.selectbox("Choose source", ("File", "Wikipedia Article"))

    # 3‑d · Upload or Wiki topic
    uploaded_file = None
    wiki_topic = None
    
    if source_type == "File":
        uploaded_file = st.file_uploader("Upload .docx / .txt / .pdf", ["docx", "txt", "pdf"])
    else:
        wiki_topic = st.text_input("Wikipedia topic")

    st.markdown("---")
    st.markdown("[GitHub Repo](https://github.com/junesaisquoi/FULLSTACK-GPT)")
            
# ─────────────────────── 4. Helpers ─────────────────────
# Combine many docs into one context string
format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)

# Cache file splitting
@st.cache_data(show_spinner="Loading file …")
def split_file(file):
    path = Path(".cache/quiz_files") / file.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file.read())

    loader = UnstructuredFileLoader(str(path))
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    return loader.load_and_split(text_splitter=splitter)

# Cache Wikipedia search
@st.cache_data(show_spinner="Searching Wikipedia …")
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(topic)

# extract function-call JSON → dict
from langchain.schema import AIMessage

def extract_quiz(message: AIMessage):
    args = message.additional_kwargs.get("function_call", {}).get("arguments", "{}")
    return json.loads(args)

# ───────────────────── 5 · Function‑calling setup ────────────────
function_schema = {
    "name": "create_quiz",
    "description": "Generate quiz questions with answers.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

# Dynamic LLM (uses user key if provided)
llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.3 if difficulty == "hard" else 0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    openai_api_key=api_key or None,
).bind(function_call={"name": "create_quiz"}, functions=[function_schema])

# Prompt that includes difficulty request
quiz_prompt = ChatPromptTemplate.from_template(
    """
    Based ONLY on the context below, create **10 {difficulty} multiple‑choice questions**. Each question
    must have 4 options with exactly one correct answer (mark the correct option with (o)).
    Context: {context}
    """
)

quiz_chain = (
    {"context": format_docs, "difficulty": lambda _: difficulty} | quiz_prompt | llm | extract_quiz
)

# ───────────────────── 6 · Prepare documents ───────────────────────
docs = None
if source_type == "File" and uploaded_file:
    docs = split_file(uploaded_file)
elif source_type == "Wikipedia Article" and wiki_topic:
    docs = wiki_search(wiki_topic)

    
# ───────────────────── 7 · Quiz generation & display ───────────────
if not docs:
    st.info("Upload a file or enter a Wikipedia topic to start.")
else:
    # Keep quiz & answers in session so user can retry without re‑querying LLM
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
        st.session_state.attempts = 0

    if st.session_state.quiz_data is None:
        st.session_state.quiz_data = quiz_chain.invoke(docs)
        st.session_state.attempts = 0

    quiz = st.session_state.quiz_data

    with st.form("quiz_form"):
        score = 0
        total = len(quiz["questions"])
        for idx, q in enumerate(quiz["questions"], 1):
            st.write(f"**Question {idx}:** {q['question']}")
            options = [a["answer"] for a in q["answers"]]
            choice = st.radio("", options, index=None, key=f"q{idx}")
            # Evaluate immediately on submit
            correct_ans = next(a["answer"] for a in q["answers"] if a["correct"])
            if choice == correct_ans:
                score += 1
        submitted = st.form_submit_button("Submit Answers")

    if submitted:
        st.session_state.attempts += 1
        if score == total:
            st.success(f"Perfect! {score}/{total} 🎉")
            st.balloons()
            # Reset so a new quiz can be generated next time
            st.session_state.quiz_data = None
        else:
            st.warning(f"You scored {score}/{total}. Try again!")
            if st.button("Retake Quiz"):
                # Clear stored answers and immediately rerun app
                for i in range(1, total + 1):
                    st.session_state.pop(f"q{i}", None)
                import streamlit as _st; _st.experimental_rerun()