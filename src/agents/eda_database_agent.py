# ✅ Optimized EDAVisualizationAgent with async readiness and caching
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tools.python_run import execute_visualizations, extract_python_code
from langchain_core.output_parsers import StrOutputParser
from tools.eda import (
    explain_data,
    generate_sweetviz_report
)
import pandas as pd

class EDAVisualizationAgent:
    def __init__(self, conn, llm, data_raw):
        self.conn = conn
        self.llm = llm
        # Convert raw data to DataFrames: support multiple tables
        if isinstance(data_raw, dict):
            self.data_raw = {tbl: pd.DataFrame(records) for tbl, records in data_raw.items()}
        else:
            self.data_raw = pd.DataFrame(data_raw)
        self.description = explain_data(self.data_raw)

    def run(self, question, decision, smart_filter_schema):
        if decision == "Agent1":
            return self.run_eda(question)
        elif decision == "Agent2":
            return self.run_visualization(question, smart_filter_schema)
        else:
            raise ValueError("Invalid decision provided: must be 'Agent1' or 'Agent2'")

    def fix_extract_python_code(self, response):
        prompt_template = """
        You are a helpful assistant that only extracts Python code from responses. 
        Here is the response:
        {response}
        
        Return only the Python code. Do not include any text or explanations. Just return the code inside the triple backticks."""

        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({"response": response})

    def fix_python_code(self, code, error, question):
        fix_prompt = PromptTemplate(
            input_variables=["code", "error", "question"],
            template="""
You are a Python code fixer AI agent.

You will be given:
- A user question
- A broken Python code snippet
- An error message from execution

Your task is to fix the code so it runs successfully and produces the expected result.

BROKEN CODE:
{code}

ERROR MESSAGE:
{error}

USER QUESTION:
{question}

INSTRUCTIONS:
- Identify the root cause of the error.
- Make sure you use only compatiable column types for the specified plots.
- Fix any syntax, logic, or compatibility issues.
- Ensure all libraries used are valid and imported.
- The code should be ready to run — no placeholders, no mockups.
- Return only the Python code. Do not include any text or explanations. Just return the code inside the triple backticks.
- Generate the visualization(s) based on the user's query. Always return the result as a **list of visualization objects** (even if it's a single visualization).
- In the code generated, do not call the function.  
- The function name must be `create_visualizations`.
- Make sure to delete all the temporary files you create in the code.

Avoid these:
- Do not include steps to save files.
- Do not include steps to modify existing tables, create new tables or modify the database schema.
- Make sure not to alter the existing data in the database.
- Make sure not to include unsafe code that could cause data loss or corruption.
"""
        )
        chain = fix_prompt | self.llm | StrOutputParser()
        return chain.invoke({"code": code, "error": str(error), "question": question})

    def run_eda(self, question):
        description_template = """
        Role:
        You are an expert data analyst helping users understand datasets clearly and simply.
        Instructions:
        - If the user asks for a downloadable report (e.g. mentions "PDF", "download", "report"), return only: SWEETVIZ
        - Otherwise:
            - Fix typos in the question if needed.
            - Describe the dataset (what it's about, columns, datatypes) without repeating file names like 'single dataset'.
            - Highlight key stats: patterns, trends, correlations, or outliers.
            - Use plain language with light technical terms (like mean, correlation), explained simply.

        User Input:
        {question}

        Dataset Description:
        {description}
        """
        description_prompt = PromptTemplate(input_variables=["question", "description"], template=description_template)
        description_chain = description_prompt | self.llm | StrOutputParser()
        response = description_chain.invoke({"question": question, "description": self.description})
        if response == "SWEETVIZ":
            return generate_sweetviz_report(self.data_raw)
        return response, None

    def run_visualization(self, question, smart_filter_schema):
        visual_template = """
        Role:
        You are a data visualization expert. Your task is to create clear and insightful visualizations that address the user's query. You must decide on the most appropriate type of chart or plot based on the query, even if it is not explicitly stated.

        Instructions:  
        - Use only the columns and tables present in the metadata.
        - Make sure you use the right table name.
        - The database may contain multiple related tables; use JOINs where necessary.
        - Use this table to retrieve the necessary data for your visualizations. You can use SQL queries to access the relevant data.  
        - Generate the visualization(s) based on the user's query. Always return the result as a **list of visualization objects** (even if it's a single visualization).
        - In the code generated, do not call the function.  
        - The function name must be `create_visualizations`.  
        - Use Python libraries such as Plotly, Matplotlib, or Seaborn to generate the visualization(s).  
        - Include code that imports the required libraries, reads the data from the SQLite database, and creates the visualization(s).  
        - Provide an explanation for the significance of the visualization in the context of the query and the insights it offers.  
        - Make sure to delete all the temporary files you create in the code.

        Output Format:  
        1. Python code that includes a `create_visualizations()` function.  
        2. The function must return a list of visualization objects.  
        3. Do not repeat the user's question or mention typos in the response.  

        Avoid these:
        - Do not include steps to save files.
        - Do not include steps to modify existing tables, create new tables or modify the database schema.
        - Make sure not to alter the existing data in the database.
        - Make sure not to include unsafe code that could cause data loss or corruption.

        User Input: {question}
        Smart Schema : {smart_filter_schema}
        """
        visual_prompt = PromptTemplate(input_variables=["question", "smart_filter_schema"], template=visual_template)
        visual_chain = visual_prompt | self.llm | StrOutputParser()
        response = visual_chain.invoke({"question": question, "smart_filter_schema": smart_filter_schema})

        pycode = extract_python_code(response, 1)
        if pycode == "NO PYTHON CODE FOUND":
            fix_pycode = self.fix_extract_python_code(response)
            pycode = extract_python_code(fix_pycode, 2)

        visualisations = execute_visualizations(pycode)

        for _ in range(10):
            if isinstance(visualisations, str) and visualisations.startswith("Error executing the code:"):
                new_pycode = self.fix_python_code(pycode, visualisations, question)
                pycode = extract_python_code(new_pycode, 1)
                if pycode == "NO PYTHON CODE FOUND":
                    fix_pycode = self.fix_extract_python_code(new_pycode)
                    pycode = extract_python_code(fix_pycode, 2)
                visualisations = execute_visualizations(pycode)
            else:
                break

        return response, visualisations
