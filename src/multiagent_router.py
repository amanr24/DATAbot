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
    async def inner(x):
        agent = EDAVisualizationAgent(conn, llm, data_raw)
        response, artifact = agent.run(x["question"], "Agent1", None)
        if artifact:
            handle_artifact(artifact)
        return response
    return RunnableLambda(lambda x: asyncio.run(inner(x)))

def viz_agent_fn(conn, data_raw, llm):
    async def inner(x):
        agent = EDAVisualizationAgent(conn, llm, data_raw)
        sql_agent = SQLDatabaseAgent(llm=llm, conn=conn)
        response, artifact = agent.run(x["question"], "Agent2", sql_agent.smart_filter_schema(x["question"]))
        if artifact:
            handle_artifact(artifact)
        return response
    return RunnableLambda(lambda x: asyncio.run(inner(x)))

def sql_agent_fn(conn, data_raw, llm):
    async def inner(x):
        sql_agent = SQLDatabaseAgent(llm=llm, conn=conn)
        sql_agent.smart_filter_schema(x["question"])
        sql_query = sql_agent.generate_sql_query(x["question"])
        result, executed_query = sql_agent.run_sql_query(sql_query)

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
    router_prompt = PromptTemplate.from_template("""
You are an intelligent agent selector for analyzing multi-table datasets.

Inputs:
- Current user message: {input}
- Chat history: {history}

Output only a Python list in this format:
["AgentX", "Rephrased question"]

Where AgentX is:
- Agent1 (🧠 Data Exploration / Statistics):
- When the user asks for high-level summaries, insights, dataset descriptions, distributions, or overall explanations.
- Examples: "Describe this dataset", "What is the structure?", "Give me a summary", "What is interesting in the data?"

- Agent2 (📊 Data Visualization):
- When the user asks to visualize or plot the data (heatmap, histogram, pie chart, trends, etc).
- Examples: "Show correlation heatmap", "Plot price trends", "Visualize monthly sales"

- Agent3 (🗄️ SQL Query):
- When the user is asking for precise calculations, aggregations, filtering, or row-level data that can be fetched using SQL queries.
- Look for keywords like: "Join", "average", "total", "count", "maximum", "minimum", "select", "filter", "how many", "retrieve", or if dates/time ranges are involved.
- Examples: "What was the average sales in March?", "Get rows where quantity > 100", "Show all orders from 2022"

If the input is already specific, do not rephrase. If the user asks what the last question was, return the string "LAST_QUESTION".
""")
    return (
        {"input": lambda x: x["question"],
         "history": lambda x: x.get("chat_history", StreamlitChatMessageHistory(key="chat")).messages if hasattr(x.get("chat_history"), "messages") else []}
        | router_prompt
        | llm
        | StrOutputParser()
    )

def get_router_with_memory(llm, conn, data_raw: dict, pdf_chains: list, chat_history: BaseChatMessageHistory):
    router_chain = get_router_chain(llm)

    def router_executor(x):
        query = x['question']
        for chain in pdf_chains or []:
            try:
                answer = chain.run(query)
                if answer and answer.strip():
                    return answer
            except Exception:
                continue

        decision, rephrased = eval(router_chain.invoke(x))
        x["question"] = rephrased
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
    if isinstance(artifact, list):
        for fig in artifact:
            st.plotly_chart(fig, use_container_width=True)
    elif isinstance(artifact, dict) and "report_path" in artifact:
        with open(artifact["report_path"], "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
        download_link = create_download_link(artifact["report_path"])
        st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.error("⚠️ Invalid artifact format.")

def create_download_link(path, filename="sweetviz_report.html"):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href
