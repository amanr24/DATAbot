'''
pdf_database_agent.py

Handles PDF text and table extraction with caching and threading, then builds a RAG index 
for question-answering over multiple PDFs using LangChain and Groq's LLM.
'''

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import pdfplumber  # PDF parsing for text and tables
import pandas as pd
import streamlit as st  # Caching decorator for Streamlit apps

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


def deduplicate_columns(columns: List[str]) -> List[str]:
    '''
    Ensure unique column names by appending suffixes to duplicates.

    Example:
        ['A', 'B', 'A'] -> ['A', 'B', 'A.1']
    '''
    seen = {}
    new_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col] - 1}")
    return new_cols


def is_valid_table(table: List[List[str]]) -> bool:
    '''
    Check if a parsed table has consistent row lengths and a header row.

    - Must have at least 2 rows
    - All non-empty rows have the same number of columns
    - Header row must contain at least two non-numeric cells
    '''
    if not table or len(table) < 2:
        return False
    # Count non-empty rows and check uniformity
    row_lengths = [len(row) for row in table if any(cell and cell.strip() for cell in row)]
    if not row_lengths or len(set(row_lengths)) > 1:
        return False
    # Validate header has at least two non-numeric entries
    header = table[0]
    non_numeric = sum(1 for cell in header if cell and not cell.replace('.', '', 1).isdigit())
    return non_numeric >= 2


def extract_text_and_tables_with_fallback(page) -> Tuple[List[str], List[pd.DataFrame]]:
    '''
    Extract text and tables from a single PDF page using two fallback strategies.

    Returns:
    - List of text blocks
    - List of valid DataFrames parsed from tables
    '''    
    text_blocks = []
    tables = []

    # Extract all text if available
    text = page.extract_text()
    if text:
        text_blocks.append(text.strip())

    # Try two table extraction strategies for robustness
    for strategy in (
        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
        {"vertical_strategy": "text",  "horizontal_strategy": "text"}
    ):
        raw_tables = page.extract_tables(strategy)
        valid = []
        for table in raw_tables:
            # Clean whitespace and filter blank rows
            cleaned = [[(cell or '').strip() for cell in row]
                       for row in table if any(cell and cell.strip() for cell in row)]
            if is_valid_table(cleaned):
                df = pd.DataFrame(cleaned[1:], columns=cleaned[0])
                df = df.dropna(how='all').reset_index(drop=True)
                valid.append(df)
        if valid:
            tables = valid
            break

    return text_blocks, tables

# ✅ Cached extraction of all pages in a PDF
@st.cache_data(show_spinner=True)
def extract_all(pdf_path: str) -> Tuple[str, pd.DataFrame]:
    '''
    Extract text and tables from every page in a PDF, in parallel.

    Returns combined text string and a single DataFrame of all tables.
    '''
    # Determine number of pages
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)

    # Parallel processing of pages
    def process_page(i):
        with pdfplumber.open(pdf_path) as pdf_inner:
            return extract_text_and_tables_with_fallback(pdf_inner.pages[i])

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_page, range(num_pages)))

    # Aggregate text and tables
    all_text = []
    all_tables = []
    for text_blocks, tables in results:
        all_text.extend(text_blocks)
        all_tables.extend(tables)

    # Combine text and dedupe columns in tables
    combined_text = "\n\n".join(all_text)
    # Ensure all tables have unique column names before concat
    for i in range(len(all_tables)):
        df = all_tables[i]
        if df.columns.duplicated().any():
            all_tables[i].columns = deduplicate_columns(df.columns)

    combined_df = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()


    return combined_text, combined_df

def build_rag_index_for_multiple(pdf_paths: List[str], api_key: str) -> RetrievalQA:
    '''
    Build a RetrievalQA chain over multiple PDFs:
    1. Extract text
    2. Split into chunks
    3. Embed with HuggingFace
    4. Store in-memory
    5. Wrap in a RetrievalQA chain using Groq LLM

    Returns:
    - RetrievalQA instance ready for .run(question)
    '''
    # Load documents with metadata
    docs = []
    for path in pdf_paths:
        text, _ = extract_all(path)
        filename = os.path.basename(path)
        docs.append(Document(page_content=text, metadata={"source": filename}))

    # Chunk documents for RAG
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create vector store and embeddings
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64}
    )
    vectorstore = InMemoryVectorStore.from_documents(chunks, embed_model)

    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=api_key  # Use env var for safety
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context (with source metadata) to answer the question:
{context}

Q: {question}
A:
"""
    )

    retriever = vectorstore.as_retriever(k=5)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain
