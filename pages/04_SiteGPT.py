# 1. Imports
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import SitemapLoader
from langchain_community.vectorstores import FAISS            # ← use community vectorstores
from langchain.schema.runnable import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

def parse_page(soup):
    for tag in ("header", "footer"):
        el = soup.find(tag)
        if el:
            el.decompose()
    return soup.get_text(" ", strip=True)

# 2. Streamlit config
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

# 3 Sidebar – key + sitemap URL
DEFAULT_URL = "https://developers.cloudflare.com/sitemap.xml"

with st.sidebar:
    openai_api_key = st.text_input(
        "Input your OpenAI API Key",
        type="password",
        placeholder="Your OpenAI API Key",
    )
    st.markdown("---")
    url = st.text_input(
        "Cloudflare sitemap URL",
        value=DEFAULT_URL,
    )
    if ".xml" not in url:
        st.error("Please write down a Sitemap URL ending in .xml.")
        st.stop()
    st.markdown("---")
    st.write("GitHub: https://github.com/junesaisquoi/FULLSTACK-GPT")

url = url.split("#", 1)[0].strip()

# Stop if no key
if not openai_api_key:
    st.warning("Please input your OpenAI API Key in the sidebar to start.")
    st.stop()    

# 4 LLM and embeddings
llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openai_api_key,
    max_tokens=500,
)

class BatchedOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        BATCH = 50
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i : i + BATCH]
            out.extend(super().embed_documents(chunk))
        return out

embedding_fn = BatchedOpenAIEmbeddings(openai_api_key=openai_api_key)

# 5 Prompts
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

# 6 Helper runnables
def get_answers(inputs: dict) -> dict:
    docs     = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    results = []
    for doc in docs:
        raw = answers_chain.invoke(
            {"question": question, "context": doc.page_content}
        ).content
        score = 0
        if "Score:" in raw:
            try:
                score = int(raw.split("Score:")[1].split()[0])
            except ValueError:
                pass
        results.append(
            {
                "answer": raw,
                "score":  score,
                "source": doc.metadata.get("source", "unknown"),
                "date":   doc.metadata.get("lastmod", "unknown"),
            }
        )
    return {"question": question, "answers": results}

def choose_answer(inputs: dict):
    answers  = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{a['answer']}\nSource:{a['source']}\nDate:{a['date']}\nScore:{a['score']}"
        for a in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})

# 7 Docs loader + retriever
def build_retriever(sitemap_url: str):
    loader_kwargs = {"parsing_function": parse_page}

    # only filter when the user left the field at the default value
    if sitemap_url.strip() == DEFAULT_URL:
        loader_kwargs["filter_urls"] = [
            r"^https://developers\.cloudflare\.com/ai-gateway/.*",
            r"^https://developers\.cloudflare\.com/vectorize/.*",
            r"^https://developers\.cloudflare\.com/workers-ai/.*",
        ]

    # otherwise custom sitemap → no filter_urls
    loader = SitemapLoader(sitemap_url, **loader_kwargs)
    loader.requests_per_second = 2

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    docs = loader.load_and_split(text_splitter=splitter)

    texts     = [doc.page_content for doc in docs]
    metadatas = [doc.metadata         for doc in docs]
    all_embs  = []
    BATCH     = 50

    for i in range(0, len(texts), BATCH):
        batch_texts = texts[i : i + BATCH]
        batch_embs  = embedding_fn.embed_documents(batch_texts)
        all_embs.extend(batch_embs)

    text_and_embs = list(zip(texts, all_embs))
    store = FAISS.from_embeddings(
        text_embeddings=text_and_embs,
        metadatas=metadatas,
        embedding=embedding_fn,
    )

    return store.as_retriever(search_kwargs={"k": 3})

retriever = build_retriever(url)

# 8 Chat UI
st.markdown(
    """
    # Cloudflare SiteGPT

    🔍 Enter your OpenAI API key and (optionally) a sitemap URL in the sidebar  
    ✔️ By default, this indexes **Cloudflare’s AI Gateway**, **Vectorize** & **Workers AI** docs  
    💬 Then type any question below to get precise, scored answers with source links!
    """
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if user_q := st.chat_input("Ask…"):
    st.session_state.messages.append({"role": "user", "content": user_q})
    st.chat_message("user").markdown(user_q)

    chain = (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    reply = chain.invoke(user_q).content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply.replace("$", r"\$"))