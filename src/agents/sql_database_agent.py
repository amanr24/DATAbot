'''
sql_database_agent.py

Defines SQLDatabaseAgent for generating and executing SQL queries on a multi-table SQLite database.
Features smart schema filtering, SQL generation via LLM, auto-fixing broken queries, and result retrieval.
'''

import json
import re
import pandas as pd
from tools.sql import get_db_metadata
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


class SQLDatabaseAgent:
    '''
    Agent to interact with a SQLite database via LLM-generated SQL.

    Capabilities:
    - Load and cache database metadata
    - Smart-filter schema based on user query
    - Generate SQL query with joins and filters
    - Auto-fix broken SQL up to 5 attempts
    - Execute query and return results
    '''
    def __init__(self, llm, conn, smart_filtering=True):
        # LLM instance for prompt invocations
        self.llm = llm
        # Active SQLAlchemy connection
        self.conn = conn
        # Toggle schema filtering via LLM
        self.smart_filtering = smart_filtering
        # Raw metadata from database (schemas, tables, columns)
        self.raw_metadata = get_db_metadata(conn)
        self.filtered_metadata = None
        self.user_query = ""

    def smart_filter_schema(self, user_query: str) -> dict:
        '''
        Filters raw_metadata to relevant tables/columns for the given user_query.
        Returns JSON-like dict with pared-down schema.
        '''
        if not self.smart_filtering:
            # Skip filtering if disabled
            self.filtered_metadata = self.raw_metadata
            return self.filtered_metadata

        # Prompt to select relevant schema parts
        filter_prompt = PromptTemplate(
            template="""
            You are an expert database assistant. The database contains multiple tables loaded from CSV files.
            Based on the user's question:

            "{user_query}"

            And the complete database metadata (in JSON):
            {metadata_json}
            GOLDEN RULE:
            - Make sure you use the right table name and use right columns for them.

            Return ONLY a JSON object with relevant schemas, tables, and columns needed to answer the question.
            - Maintain the structure: Connection_url  schemas → tables → columns.
            - If some columns in a relevant table are not needed, you can still keep them if you aren't sure.
            - Include all tables/columns implied by joins.
            - Do not include extra text.
            Avoid these:
            - Do not include steps to save files.
            - Do not include steps to modify existing tables, create new tables or modify the database schema.
            - Make sure not to alter the existing data in the database.
            """,
            input_variables=["user_query", "metadata_json"]
        )

        chain = filter_prompt | self.llm | JsonOutputParser()
        self.filtered_metadata = chain.invoke({
            "user_query": user_query,
            "metadata_json": json.dumps(self.raw_metadata)
        })
        return self.filtered_metadata

    def get_filtered_metadata(self):
        '''Return filtered metadata or raw if none.'''        
        return self.filtered_metadata or self.raw_metadata

    def generate_sql_query(self, user_query: str) -> str:
        '''
        Generates an SQL query for the user_question based on filtered_metadata.
        Returns the query wrapped in a ```sql ... ``` block.
        '''
        self.user_query = user_query
        metadata = self.get_filtered_metadata()

        # Prompt LLM to generate SQL, including joins across tables if needed
        sql_prompt = PromptTemplate(
            input_variables=["user_query", "metadata_json"],
            template="""
You are a SQL expert. The database contains multiple tables loaded from CSV.
Given the database metadata and the user question, write an appropriate SQL query.

USER QUESTION:
{user_query}

DATABASE METADATA (JSON):
{metadata_json}

GUIDELINES:
- Correct typos in the question and column names when possible.
- Ensure compatibility with SQLite.
- Use only the columns and tables present in the metadata.
- Make sure you use the right table name.
- The database may contain multiple related tables; use JOINs where necessary.
- Do NOT add comments or explanations.

Return:
            - The SQL code in ```sql ``` format to collect the data and process it according to the user instructions.
            
            Avoid these:
            - Do not include steps to save files.
            - Do not include steps to modify existing tables, create new tables or modify the database schema.
            - Make sure not to alter the existing data in the database.
            - Make sure not to include unsafe code that could cause data loss or corruption.
"""
        )

        chain = sql_prompt | self.llm | StrOutputParser()
        sql_query = chain.invoke({
            "user_query": user_query,
            "metadata_json": json.dumps(metadata)
        })
        return sql_query

    def fix_sql_database_code(self, sql_query: str, error: Exception) -> str:
        '''
        Attempts to fix a broken SQL query based on the given error.
        Returns corrected SQL code only.
        '''
        fix_prompt = PromptTemplate(
            input_variables=["sql_query", "error", "user_query", "metadata_json"],
            template="""
You are a SQL fixer for SQLite databases containing multiple CSV tables.
Fix the following SQL query based on the error:

BROKEN QUERY:
{sql_query}

ERROR:
{error}

USER QUESTION:
{user_query}

METADATA:
{metadata_json}

GUIDELINES:
- Carefully look at the last error, try to return the correct the sql query.
- Remove unnecessary code blocks or delimiters like ```sql, ```, or backticks (`).
- Provide only the corrected SQL query without any extra text or explanation.
- Fix syntax or compatibility issues.
- If a join is intended, ensure the correct JOIN syntax and table aliases.
"""
        )

        chain = fix_prompt | self.llm | StrOutputParser()
        fixed_sql = chain.invoke({
            "sql_query": sql_query,
            "error": str(error),
            "user_query": self.user_query,
            "metadata_json": json.dumps(self.get_filtered_metadata())
        })
        return fixed_sql

    def run_sql_query(self, sql_query: str):
        '''
        Executes the SQL query, auto-fixing errors up to 5 retries.
        Returns (DataFrame, executed_query) or error string.
        '''
        # Extract SQL from ```sql``` block
        match = re.search(r"```sql\s*(.*?)\s*```", sql_query, re.DOTALL)
        query = match.group(1).strip() if match else sql_query.strip()

        # Try executing and auto-fix on failure
        for attempt in range(5):
            try:
                result = pd.read_sql_query(query, self.conn)
                return result, query
            except Exception as e:
                # Fix and update query
                sql_query = self.fix_sql_database_code(query, e)
                match = re.search(r"```sql\s*(.*?)\s*```", sql_query, re.DOTALL)
                match1 = re.search(r"```\n(.*?)```", sql_query, re.DOTALL)
                if match:
                    query = match.group(1).strip() if match else sql_query.strip()
                elif match1:
                    query = match1.group(1).strip() if match else sql_query.strip()
                else:
                    query = self.fix_sql_database_code(query, e)

        # Final attempt
        try:
            result = pd.read_sql_query(query, self.conn)
            return result, query
        except Exception as final_error:
            # Return error message if still failing
            return f"Final SQL error: {final_error}"
