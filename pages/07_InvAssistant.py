import streamlit as st
import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (optional local dev)
load_dotenv()

# API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Investor Assistant", page_icon="📈")

with st.sidebar:
    if not openai_api_key:
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    st.write("GitHub: https://github.com/junesaisquoi/FULLSTACK-GPT")
    st.markdown("---")

if not openai_api_key:
    st.warning("Please enter your OpenAI API Key.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# API functions
def get_ticker(inputs):
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={inputs['company_name']}&apikey={alpha_vantage_api_key}"
    response = requests.get(url).json()
    matches = response.get("bestMatches", [])
    return {"ticker": matches[0]["1. symbol"]} if matches else {"error": "Ticker not found."}

def get_income_statement(inputs):
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={inputs['ticker']}&apikey={alpha_vantage_api_key}"
    response = requests.get(url).json()
    return response.get("annualReports", {"error": "No income statement data."})

def get_balance_sheet(inputs):
    url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={inputs['ticker']}&apikey={alpha_vantage_api_key}"
    response = requests.get(url).json()
    return response.get("annualReports", {"error": "No balance sheet data."})

def get_daily_stock_performance(inputs):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={inputs['ticker']}&apikey={alpha_vantage_api_key}"
    response = requests.get(url).json()
    return response.get("Time Series (Daily)", {"error": "No stock price data."})

# Tools for Assistants v2
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Get ticker symbol from company name.",
            "parameters": {
                "type": "object",
                "properties": {"company_name": {"type": "string"}},
                "required": ["company_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Get income statement data for ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Get balance sheet data for ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": "Get daily stock price data.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        }
    }
]

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "📈 Hello! I am Investor Assistant. Enter a company name and I will analyze whether it's a good investment or not."}
    ]

if "assistant_id" not in st.session_state:
    assistant = client.assistants.create(
        name="Investor Assistant",
        instructions="You analyze companies and recommend whether to invest or not based on financials and stock performance.",
        model="gpt-4o-mini",
        tools=tools
    )
    st.session_state.assistant_id = assistant.id

if "thread_id" not in st.session_state:
    thread = client.threads.create()
    st.session_state.thread_id = thread.id

# Streamlit chat UI
st.title("📊 Investor Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me about any company"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    client.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=user_input
    )

    run = client.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=st.session_state.assistant_id
    )

    while True:
        run_status = client.threads.runs.retrieve(
            thread_id=st.session_state.thread_id,
            run_id=run.id
        )

        if run_status.status == "requires_action":
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []

            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if fn_name == "get_ticker":
                    output = get_ticker(args)
                elif fn_name == "get_income_statement":
                    output = get_income_statement(args)
                elif fn_name == "get_balance_sheet":
                    output = get_balance_sheet(args)
                elif fn_name == "get_daily_stock_performance":
                    output = get_daily_stock_performance(args)
                else:
                    output = {"error": "Unknown function."}

                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(output)
                })

            client.threads.runs.submit_tool_outputs(
                thread_id=st.session_state.thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

        elif run_status.status == "completed":
            break

    messages = client.threads.messages.list(thread_id=st.session_state.thread_id)
    response = messages.data[0].content[0].text.value
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
