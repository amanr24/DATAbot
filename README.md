<!-- README.md -->

# All-in-One Data Chatbot

A Streamlit-based data assistant that lets you upload CSVs and ask natural-language questions about your data.  
Under the hood it uses a multi-agent setup (exploration, visualization, SQL) powered by Groq’s LLM.

## 🚀 Features

- **Upload & Persist** CSVs in a local SQLite database  
- **Agent1**: Data exploration & narrative summaries (via `eda.py` & Sweetviz)  
- **Agent2**: Automatic chart generation (Plotly/Matplotlib/Seaborn)  
- **Agent3**: SQL query generation & execution (SQLAlchemy + LLM)  
- Downloadable Sweetviz HTML reports  
- Clean, interactive UI with Streamlit  

## ⚙️ Installation

```bash
git clone https://github.com/your-username/data-chatbot.git
cd data-chatbot
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
pip install -r requirements.txt
