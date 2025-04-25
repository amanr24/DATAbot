''' 
multiagent_agent.py
Implements a multi-agent routing framework for the Data Chatbot, selecting 
between EDA, Visualization, and SQL agents based on user input, with Streamlit 
for UI integration and message history support. 
'''

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

import streamlit as st
import streamlit.components.v1 as components
import base64
import pandas as pd
import asyncio

from agents.eda_database_agent import EDAVisualizationAgent
from agents.sql_database_agent import SQLDatabaseAgent


# ---------- Agent Logic ----------

def eda_agent_fn(conn, data_raw, llm):
    '''
    Returns a RunnableLambda that runs the EDA agent (Agent1) to produce narrative summaries.
    Parameters:
    - conn: active SQLAlchemy connection
    - data_raw: raw data as dict/list for EDAVisualizationAgent
    - llm: initialized LLM instance
    Returns:
    - RunnableLambda: wraps async call to EDA agent
    '''
    async def inner(x):
        # Instantiate EDA agent and run with decision "Agent1"
        agent = EDAVisualizationAgent(conn, llm, data_raw)
        response, artifact = agent.run(x["question"], "Agent1", None)
        if artifact:
            handle_artifact(artifact)
        return response

    # Wrap async function so it can be invoked synchronously
    return RunnableLambda(lambda x: asyncio.run(inner(x)))


def viz_agent_fn(conn, data_raw, llm):
    '''
    Returns a RunnableLambda that runs the Visualization agent (Agent2).
    Parameters are same as eda_agent_fn
    The agent will first perform EDA description then generate plots via SQLDatabaseAgent for schema filtering.
    '''
    async def inner(x):
        # Instantiate visualization agent and SQL schema filter
        agent = EDAVisualizationAgent(conn, llm, data_raw)
        sql_agent = SQLDatabaseAgent(llm=llm, conn=conn)
        # Agent2 decision uses smart_filter_schema to narrow metadata
        response, artifact = agent.run(x["question"], "Agent2", sql_agent.smart_filter_schema(x["question"]))
        if artifact:
            handle_artifact(artifact)
        return response

    return RunnableLambda(lambda x: asyncio.run(inner(x)))


def sql_agent_fn(conn, data_raw, llm):
    '''
    Returns a RunnableLambda that runs the SQL agent (Agent3) to generate and execute SQL queries.
    Displays generated query and results in Streamlit.
    '''
    async def inner(x):
        # Initialize SQL agent and filter schema by query
        sql_agent = SQLDatabaseAgent(llm=llm, conn=conn)
        sql_agent.smart_filter_schema(x["question"])
        # Generate the SQL query text
        sql_query = sql_agent.generate_sql_query(x["question"])
        # Execute the query, returns (result_df or message, executed_query)
        result, executed_query = sql_agent.run_sql_query(sql_query)

        # Display query and results
        st.markdown("**🧠 Generated SQL Query:**")
        st.code(executed_query, language="sql")
        if isinstance(result, pd.DataFrame):
            st.markdown("**📈 Query Result:**")
            st.dataframe(result, use_container_width=True)
        else:
            st.warning(result)

        return result if isinstance(result, str) else "Query executed successfully."

    return RunnableLambda(lambda x: asyncio.run(inner(x)))


# ---------- Router Prompt + Dispatcher ----------

def get_router_chain(llm):
    '''
    Builds and returns the routing chain that selects which agent to invoke.
    Uses a PromptTemplate to classify user input into Agent1, Agent2, or Agent3,
    and optionally rephrase the question if needed.
    Returns:
    - A Runnable chain: input -> template -> llm -> parser -> [decision, rephrased]
    '''
    router_prompt = PromptTemplate.from_template("""
You are an intelligent agent selector for analyzing multi-table datasets.
Inputs:
- Current user message: {input}
- Chat history: {history}
Output only a Python list in this format:
["AgentX", "Rephrased question"]
... (instructions omitted for brevity) ...
If the input is already specific, do not rephrase. If the user asks what the last question was, return the string "LAST_QUESTION".
""")

    # Chain definition: map input fields, apply prompt, LLM, and parse result
    return (
        {"input": lambda x: x["question"],
         "history": lambda x: x.get("chat_history", StreamlitChatMessageHistory(key="chat")).messages if hasattr(x.get("chat_history"), "messages") else []}
        | router_prompt
        | llm
        | StrOutputParser()
    )


def get_router_with_memory(llm, conn, data_raw: dict, pdf_chains: list, chat_history: BaseChatMessageHistory):
    '''
    Constructs the multi-agent chain with memory and PDF fallback.
    - Tries each pdf_chain first for PDF-based answers.
    - Uses router_chain to decide Agent1/2/3 and invokes the respective function.
    - Wraps everything in a RunnableWithMessageHistory to maintain chat context.
    '''
    router_chain = get_router_chain(llm)

    def router_executor(x):
        query = x['question']
        # PDF chains take priority
        for chain in pdf_chains or []:
            try:
                answer = chain.run(query)
                if answer and answer.strip():
                    return answer
            except Exception:
                continue

        # Decide on agent and rephrase question
        decision, rephrased = eval(router_chain.invoke(x))
        x["question"] = rephrased
        # Dispatch to appropriate agent
        if decision == "Agent1":
            return eda_agent_fn(conn, data_raw, llm).invoke(x)
        elif decision == "Agent2":
            return viz_agent_fn(conn, data_raw, llm).invoke(x)
        elif decision == "Agent3":
            return sql_agent_fn(conn, data_raw, llm).invoke(x)
        else:
            raise ValueError(f"Unknown agent decision: {decision}")

    multi_agent_chain = RunnableLambda(router_executor)
    return RunnableWithMessageHistory(
        multi_agent_chain,
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


# ---------- Helpers ----------

def handle_artifact(artifact):
    '''
    Renders agent artifacts in Streamlit: 
    - Lists of plotly figures
    - Sweetviz report HTML with download link
    '''
    if isinstance(artifact, list):
        # Plot each figure
        for fig in artifact:
            st.plotly_chart(fig, use_container_width=True)
    elif isinstance(artifact, dict) and "report_path" in artifact:
        # Load and embed HTML report
        with open(artifact["report_path"], "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
        download_link = create_download_link(artifact["report_path"])
        st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.error("⚠️ Invalid artifact format.")


def create_download_link(path, filename="sweetviz_report.html"):
    '''
    Generates a base64 download link for a file in Streamlit.
    Parameters:
    - path: local file path
    - filename: download name
    Returns:
    - HTML <a> tag string
    '''
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href
