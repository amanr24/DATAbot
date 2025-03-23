import sqlite3
from typing import List, Tuple
from langchain.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Initialize Ollama model once
llm = Ollama(model="llama3")

# Global variable to store schema
SCHEMA_STR = ""

# Define PromptTemplate once globally
PROMPT_TEMPLATE = PromptTemplate.from_template(
    """You are an expert in SQLite. Generate only valid SQLite queries, ensuring compatibility with SQLite's syntax.  
Do not use unsupported features such as `MEDIAN()`, `STORED PROCEDURE`, or `SEQUENCES`.  
Follow these constraints:  
- Use `WITH` clauses for complex queries.  
- Use `LIMIT` and `OFFSET` for pagination.  
- Use `CASE` for conditional logic.  
- Avoid `RIGHT JOIN` and `FULL OUTER JOIN` (since SQLite does not support them).  
- Use `strftime('%Y-%m-%d', date_column)` for date formatting.
    The database contains a table named `{table_name}` with the following schema:

    {schema}

    Generate an sqlite query to answer: "{question}"
    Use the exact table name `{table_name}` in the query.
    Only return the sqlite query, no explanation.
    """
)


def fetch_schema():
    """Fetches the schema once and stores it in a global variable."""
    global SCHEMA_STR
    with sqlite3.connect("uploaded_csv_sql.db") as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(table_name)")
        schema_info = cursor.fetchall()
    
    # Generate schema string
    SCHEMA_STR = "\n".join([f"{col[1]} ({col[2]})" for col in schema_info])
    print("Schema Cached: ", SCHEMA_STR)


# Call this function once to cache the schema
fetch_schema()


class GeneratingResponse:
    def __init__(self, chatbot: List, message: str) -> None:
        self.chatbot = chatbot
        self.message = message

    def generate_sql_query(self):
        """Generates an sqlite query using the cached schema and user question."""
        print("Generating SQL query...")

        prompt = PROMPT_TEMPLATE.format(
            schema=SCHEMA_STR, table_name="table_name", question=self.message
        )
        response = llm.invoke(prompt)
        return response.strip()

    def execute_query(self, sql_query):
        """Executes the generated SQL query on SQLite and returns the result."""
        with sqlite3.connect("uploaded_csv_sql.db") as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql_query)
                result = cursor.fetchall()
                return result
            except Exception as e:
                return f"❌ Error executing query: {e}"

    def run(self):
        """Runs the query generation and execution pipeline."""
        generated_query = self.generate_sql_query()
        print("Generated SQL Query:", generated_query)

        query_result = self.execute_query(generated_query)

        self.chatbot.append(
            (self.message, f"📝 **SQL Query:**\n```sql\n{generated_query}\n```\n📊 **Result:** {query_result}")
        )
        return "", self.chatbot


class ChatAgent:
    """
    A ChatBot class capable of responding to messages using SQL queries.
    """
    
    @staticmethod
    def respond(chatbot: List, message: str) -> Tuple:
        """Handles user query using the cached schema."""
        input_txt, chatbot = GeneratingResponse(chatbot, message).run()
        return input_txt, chatbot
