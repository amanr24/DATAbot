# optimized_sql_agent.py
import json
import pandas as pd
import re
from tools.sql import get_db_metadata
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

class SQLDatabaseAgent:
    def __init__(self, llm, conn, smart_filtering=True):
        self.llm = llm
        self.conn = conn
        self.smart_filtering = smart_filtering
        self.raw_metadata = get_db_metadata(conn)
        self.filtered_metadata = None
        self.user_query = ""

    def smart_filter_schema(self, user_query):
        if not self.smart_filtering:
            self.filtered_metadata = self.raw_metadata
            return self.filtered_metadata
        print("userquery = ", user_query)
        print("self.raw_metadata = ", self.raw_metadata)

        # Filter metadata to relevant tables/columns for the query
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
        print("self.filtered_metadata = ", self.filtered_metadata)

        return self.filtered_metadata

    def get_filtered_metadata(self):
        return self.filtered_metadata or self.raw_metadata

    def generate_sql_query(self, user_query):
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

    def fix_sql_database_code(self, sql_query, error):
        # Fix broken SQL queries by prompting LLM
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

    def run_sql_query(self, sql_query):
        # Extract SQL from ```sql``` block
        match = re.search(r"```sql\s*(.*?)\s*```", sql_query, re.DOTALL)
        query = match.group(1).strip() if match else sql_query.strip()
        print("query : ",query)

        # Try executing and auto-fix on failure
        for attempt in range(5):
            try:
                print("\n\n----------------------------------------")
                result = pd.read_sql_query(query, self.conn)
                print("try result = ",result )
                print("try query = ",query)

                return result, query
            except Exception as e:
                print("\n\nerror = ",e)
                sql_query = self.fix_sql_database_code(query, e)
                match = re.search(r"```sql\s*(.*?)\s*```", sql_query, re.DOTALL)
                match1 = re.search(r"```\n(.*?)```", sql_query, re.DOTALL)
                if match:
                    query = match.group(1).strip() if match else sql_query.strip()
                elif match1:
                    query = match1.group(1).strip() if match else sql_query.strip()
                else:
                    query = self.fix_sql_database_code(query, e)
                print("fixed query = ", query)

        # Final attempt
        try:
            result = pd.read_sql_query(query, self.conn)
            print("try final result = ",result)
            print("try final query = ",query)
            return result, query
        except Exception as final_error:
            return f"Final SQL error: {final_error}"


