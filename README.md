[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)]
[![Issues](https://img.shields.io/github/issues/your-username/data-chatbot)](https://github.com/your-username/data-chatbot/issues)
[![Stars](https://img.shields.io/github/stars/your-username/data-chatbot?style=social)](https://github.com/your-username/data-chatbot/stargazers)

# DATAbot

A Streamlit-based data assistant that lets you upload CSVs and PDFs, then ask natural-language questions about your data.  
Under the hood it uses a multi-agent setup (EDA summaries, visualizations, SQL, and RAG over PDFs) powered by Groq’s LLM.
A powerful dashboard is also included for AutoML, forecasting, and simulation.

---


## 🌐 Live App

Try the live version of DATAbot on Hugging Face Spaces:

👉 [**Live Demo**](https://huggingface.co/spaces/amanr24/DATAbot)

> No installation needed — upload your files and chat with your data right from your browser.

---

## 🚀 Features

- **ChatBot Mode**:
  - **Upload & Persist CSVs** in a local SQLite database  
  - **Agent1**: Data exploration & narrative summaries (`eda.py` + Sweetviz)  
  - **Agent2**: Automatic chart generation (Plotly / Matplotlib / Seaborn)  
  - **Agent3**: SQL query generation & execution (SQLAlchemy + LLM)  
  - **RAG over PDFs**: Question-answering on multiple PDF documents  
  - **Downloadable Sweetviz HTML reports**  
  - **Multilingual voice I/O**: Whisper-based STT + browser TTS playback  
  - Clean, responsive UI with Streamlit
- **Dashboard Mode**: Switch to an interactive dashboard for:
  - Data Cleaning (auto, fill, drop, custom)
  - AutoML with PyCaret (classification & regression)
  - What-If Simulation
  - Forecast Playground with Prophet
  - Downloadable EDA and cleaned datasets


---

## 📸 Screenshot

![App Screenshot](/images/app.png)

---

## ⚙️ Installation

```bash
git clone https://github.com/amanr24/DATAbot.git
cd DATAbot
pip install -r requirements.txt
```

---

### ▶️ Usage

```bash
streamlit run src/app.py
```

1. Enter your **[Groq API key](https://console.groq.com/keys)** in the sidebar to initialize the chatbot.
2. Upload one or more **CSV** files for structured analysis.
3. Optionally upload **PDF** documents to enable PDF question answering.
4. Choose between:
   - 🧠 **Chatbot Mode** — Ask questions in plain English (or voice):
     - Get dataset summaries, visualizations, or auto-generated SQL queries.
     - View responses and plots inline.
     - Responses are read aloud in your language.
   - 📊 **Dashboard Mode** — Launch an interactive panel for:
     - Data cleaning and transformation
     - Automated model building with PyCaret (classification & regression)
     - What-If simulation with live predictions
     - Time series forecasting with Prophet
     - EDA report downloads and performance metrics
5. Enjoy the insights — all without writing a single line of code!

---

## 📁 Project Structure

```
.
├── src/
│   ├── app.py                      # Main Streamlit app
│   ├── multiagent_router.py        # Agent routing logic
├── ├── tools/
│   ├── ├── python_run.py               # Code-execution & extraction utils
│   ├── ├── eda.py                      # EDA tools (narrative + Sweetviz)
│   ├── ├── sql.py                      # DB metadata extraction
│   ├── ├── dataframe.py                # DataFrame summary helper
│   ├── ├── voice_translator.py         # Whisper STT + translation
│   ├── ├── tts_renderer.py             # Browser TTS integration
│   └── └── dashboard.py                # Dashboard integration
├── ├── agents/
│   ├── ├── eda_database_agent.py       # EDA & Visualization agent class
│   ├── ├── sql_database_agent.py       # SQL agent (generation + execution)
│   └── └── pdf_database_agent.py       # PDF RAG chain builder
├── .images/
│   └── app.png              # UI preview
├── requirements.txt
├── .gitignore
├── LICENSE
```

---

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).  

---

## Acknowledgements
- Built with Streamlit
- Powered by LangChain and Groq
- Thanks to the contributors and the open-source community for their support.
