# InvAssistant.py (Alpha Vantage Version - Corrected)

import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import requests

# Load environment variables
load_dotenv()

# Alpha Vantage API Key from .env or Streamlit secrets
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Investor Assistant", page_icon="📈")

with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    st.write("GitHub: https://github.com/junesaisquoi/FULLSTACK-GPT")
    st.markdown("---")

if not openai_api_key:
    st.warning("Please enter your OpenAI API Key in sidebar to start.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# Functions using Alpha Vantage endpoints
def get_ticker(inputs):
    company_name = inputs['company_name']
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={alpha_vantage_api_key}"
    response = requests.get(url)
    data = response.json()
    matches = data.get("bestMatches", [])
    if matches:
        return {"ticker": matches[0]["1. symbol"]}
    else:
        return {"error": f"Could not find ticker symbol for {company_name}"}

def get_income_statement(inputs):
    ticker = inputs['ticker']
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={alpha_vantage_api_key}"
    response = requests.get(url)
    data = response.json()
    if "annualReports" in data:
        return data["annualReports"]
    else:
        return {"error": "No income statement data found."}

def get_balance_sheet(inputs):
    ticker = inputs['ticker']
    url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={alpha_vantage_api_key}"
    response = requests.get(url)
    data = response.json()
    if "annualReports" in data:
        return data["annualReports"]
    else:
        return {"error": "No balance sheet data found."}

def get_daily_stock_performance(inputs):
    ticker = inputs['ticker']
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={alpha_vantage_api_key}"
    response = requests.get(url)
    data = response.json()
    if "Time Series (Daily)" in data:
        return data["Time Series (Daily)"]
    else:
        return {"error": "No stock price data found."}

# Define tools/functions for OpenAI assistant
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Get ticker symbol from company name",
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
            "description": "Get income statement data for ticker",
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
            "description": "Get balance sheet data for ticker",
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
            "description": "Get daily stock price data",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        }
    }
]

# Streamlit chat logic
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "📈 Hello! I am Investor Assistant. Enter a company name and I will analyze whether it's a good investment or not."}
    ]

if "assistant_id" not in st.session_state:
    assistant = client.beta.assistants.create(
        name="Investor Assistant",
        instructions="You analyze companies and recommend whether to invest or not based on financials and stock performance.",
        model="gpt-4o-mini",
        tools=tools
    )
    st.session_state.assistant_id = assistant.id

if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

st.title("📊Investor Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me about any company"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=user_input
    )

    run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=st.session_state.assistant_id
    )

    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=st.session_state.thread_id,
            run_id=run.id
        )

        if run_status.status == "requires_action":
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                if function_name == "get_ticker":
                    output = get_ticker(arguments)
                elif function_name == "get_income_statement":
                    output = get_income_statement(arguments)
                elif function_name == "get_balance_sheet":
                    output = get_balance_sheet(arguments)
                elif function_name == "get_daily_stock_performance":
                    output = get_daily_stock_performance(arguments)
                else:
                    output = {"error": "Unknown function"}

                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(output)
                })

            client.beta.threads.runs.submit_tool_outputs(
                thread_id=st.session_state.thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
        elif run_status.status == "completed":
            break

    messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
    response = messages.data[0].content[0].text.value
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
