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

# ───────────────────── 5 · LLM factory ──────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm(key: str, level: str):
    """Return a ChatOpenAI bound to the create_quiz function.
    If no key provided yet, halt execution so the rest of the script
    (which relies on `llm`) is not evaluated.
    """
    if not key:
        st.info("Enter your OpenAI API key to enable quiz generation.")
        st.stop()
    return (
        ChatOpenAI(
            model="gpt-3.5-turbo-1106",
            temperature=0.3 if level == "hard" else 0.1,
            streaming=True,
            openai_api_key=key,
        ).bind(function_call={"name": "create_quiz"}, functions=[function_schema])
    )

# Build the LLM **before** it is referenced elsewhere
llm = get_llm(api_key, difficulty)

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
    # Cache quiz so user can retry without calling the LLM again
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None

    if st.session_state.quiz_data is None:
        st.session_state.quiz_data = quiz_chain.invoke(docs)
        # Clear any previous answers
        for k in list(st.session_state.keys()):
            if k.startswith("q"):
                st.session_state.pop(k)

    quiz = st.session_state.quiz_data

    # ── Build the form with radios ──
    with st.form("quiz_form"):
        for idx, q in enumerate(quiz["questions"], 1):
            st.write(f"**Question {idx}:** {q['question']}")
            options = [a["answer"] for a in q["answers"]]
            st.radio("", options, index=None, key=f"q{idx}")
        submitted = st.form_submit_button("Submit Answers")

    # ── Feedback after submission ──
    if submitted:
        score = 0
        total = len(quiz["questions"])
        for idx, q in enumerate(quiz["questions"], 1):
            user_ans = st.session_state.get(f"q{idx}")
            correct_ans = next(a["answer"] for a in q["answers"] if a["correct"])
            if user_ans == correct_ans:
                st.success(f"Question {idx}: Correct ✓")
                score += 1
            else:
                st.error(f"Question {idx}: Wrong ✗ — correct: {correct_ans}")
        st.info(f"Your score: {score}/{total}")

        # Perfect score ➜ balloons + new quiz next round
        if score == total:
            st.balloons()
            st.session_state.quiz_data = None
        else:
            # Retake button clears answers only
            if st.button("Retake Quiz"):
                for idx in range(1, total + 1):
                    st.session_state.pop(f"q{idx}", None)
                st.experimental_rerun()
