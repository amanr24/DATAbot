'''
app.py
Main Streamlit application for DataBot. Provides a chat-based data assistant and a dashboard mode.
Supports CSV uploads, PDF QA via RAG, multi-agent query routing (EDA, Visualization, SQL), and TTS.
'''

import os
import re
import tempfile
import asyncio

import streamlit as st
import pandas as pd
import sqlalchemy as sql
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from multiagent_router import get_router_with_memory
from tools.voice_translator import VoiceTranslator
from tools.tts_renderer import TextToSpeechRenderer
from tools.dashboard import render_dashboard
from agents.pdf_database_agent import build_rag_index_for_multiple

# Constants for DB path and session key
DB_PATH = "uploaded_csv_db.db"
SESSION_ID = "chat"

# Configure Streamlit page
st.set_page_config(page_title="DataBot", layout="wide")
st.title("🤖 DataBot - Ask Anything About Your Data")

# Sidebar: configuration inputs and file upload
with st.sidebar:
    st.markdown("## Configuration")
    mode = st.selectbox("Select Mode:", ("Chatbot", "Dashboard"), key="mode_toggle")
    api_key = st.text_input("Enter your API Key:", type="password")
    csv_files = st.file_uploader("Upload CSVs", type="csv", accept_multiple_files=True)
    pdf_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    # 'New Chat' resets cache and session state
    if st.button("New Chat"):
        st.cache_data.clear()
        st.session_state.clear()

# Initialize session state defaults
st.session_state.setdefault("messages", [])
st.session_state.setdefault("data_raw", {})
st.session_state.setdefault("pdf_chain", None)
st.session_state.setdefault("pdf_paths", [])
st.session_state.setdefault("pdf_temp_paths", [])
st.session_state.setdefault("chat_history", StreamlitChatMessageHistory(key=SESSION_ID))

@st.cache_data(show_spinner=False)
def upload_csv_to_sqlite(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    '''
    Store a DataFrame to SQLite under the given table name.
    Normalizes column names and caches result.
    Returns the DataFrame for display.
    '''
    # Clean up column names
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    # Create engine and write to SQLite
    engine = sql.create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    return df


def handle_user_query(user_input: str, audio, translator: VoiceTranslator, router):
    '''
    Process the user's input: translate audio, route through agents or RAG chain,
    render code blocks, translate back, and speak via TTS.
    - user_input: raw text if no audio
    - audio: uploaded audio bytes or None
    - translator: VoiceTranslator instance for STT/translation
    - router: multi-agent router runnable
    '''
    detected_lang = "en"

    # If audio provided, convert to English text
    if audio:
        user_input = translator.audio_to_english(audio)
        detected_lang = translator.get_detected_language()

    # Append user message to history and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            try:
                # If a PDF RAG chain exists, try answering via PDF first
                if st.session_state['pdf_chain']:
                    ans = st.session_state['pdf_chain'].run(user_input)
                    response = ans.strip() or router.invoke(
                        {'question': user_input},
                        {'configurable': {'session_id': SESSION_ID}}
                    )
                else:
                    # Otherwise route to EDA/Viz/SQL agents
                    response = router.invoke(
                        {'question': user_input},
                        {'configurable': {'session_id': SESSION_ID}}
                    )

                # Extract and display any Python code blocks
                for code in re.findall(r"```(?:python)?\n(.*?)```", response, flags=re.DOTALL):
                    st.code(code.strip(), language="python")

                # Strip code blocks for text-only output
                text_only = re.sub(r"```.*?```", "", response, flags=re.DOTALL)

                # If audio, convert response back to original language
                if audio:
                    text_only = translator.english_to_original_language(text_only)
                    detected_lang = translator.get_detected_language()

                # Map detected language to TTS locale
                lang_map = {
                    "en": "en-US", "hi": "hi-IN", "fr": "fr-FR",
                    "de": "de-DE", "es": "es-ES", "ar": "ar-SA",
                    "ja": "ja-JP", "it": "it-IT"
                }
                tts_lang = lang_map.get(detected_lang, f"{detected_lang}-IN")
                tts = TextToSpeechRenderer(lang=tts_lang)
                # Render spoken response
                tts.render(text_only, f"msg_{len(st.session_state.messages)}")

                # Append assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": text_only})

            except Exception as e:
                st.error(f"❌ Error: {e}")


# Only proceed if API key supplied
if api_key:
    # Initialize LLM and DB connection
    llm = ChatGroq(model="llama3-70b-8192", api_key=api_key)
    sql_engine = sql.create_engine(f"sqlite:///{DB_PATH}")
    conn = sql_engine.connect()

    # Handle CSV uploads: load each into SQLite and session state
    for file in csv_files or []:
        table_name = os.path.splitext(file.name)[0].replace(" ", "_")
        df = pd.read_csv(file)
        df = upload_csv_to_sqlite(df, table_name)
        st.success(f"✅ {table_name} loaded into SQLite!")
        st.dataframe(df.head(), use_container_width=True)
        st.session_state["data_raw"][table_name] = df.to_dict(orient="records")

    # Handle PDF uploads: build or refresh RAG index
    if pdf_files:
        names = [f.name for f in pdf_files]
        if st.session_state["pdf_chain"] is None or names != st.session_state["pdf_paths"]:
            tmp_paths = []
            for pdf in pdf_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(pdf.getbuffer()); tmp.close()
                tmp_paths.append(tmp.name)
            st.session_state["pdf_chain"] = build_rag_index_for_multiple(tmp_paths, api_key)
            st.session_state["pdf_paths"] = names
            st.session_state["pdf_temp_paths"] = tmp_paths
            st.success(f"{len(tmp_paths)} PDF(s) indexed for QA")

    # Chatbot mode: render messages and prompt for input
    if mode == "Chatbot":
        audio = st.sidebar.audio_input("🎤 Speak your question")
        translator = VoiceTranslator()
        router = get_router_with_memory(
            llm=llm,
            conn=conn,
            data_raw=st.session_state["data_raw"],
            pdf_chains=[st.session_state["pdf_chain"]] if st.session_state["pdf_chain"] else [],
            chat_history=st.session_state["chat_history"]
        )

        # Replay past messages
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    tts = TextToSpeechRenderer()
                    tts.render(msg["content"], f"msg_{i}")
                else:
                    st.markdown(msg["content"])

        # Get new user input
        user_input = st.chat_input("Type your message...")
        if user_input or audio:
            handle_user_query(user_input, audio, translator, router)

    # Dashboard mode: render a Pandas DataFrame dashboard
    elif mode == "Dashboard":
        if st.session_state['data_raw']:
            first = next(iter(st.session_state['data_raw']))
            df_dash = pd.DataFrame(st.session_state['data_raw'][first])
        else:
            df_dash = pd.DataFrame()
        render_dashboard(df_dash)

# Prompt to enter API key if missing
else:
    st.warning("Please enter your API key in the sidebar to start using the chatbot.")
