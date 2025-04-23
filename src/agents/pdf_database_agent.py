# ✅ Optimized trial.py with caching and threading for PDF extraction
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import pdfplumber
import pandas as pd
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


def deduplicate_columns(columns):
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


def is_valid_table(table: list) -> bool:
    if not table or len(table) < 2:
        return False
    row_lengths = [len(row) for row in table if any(cell and cell.strip() for cell in row)]
    if not row_lengths or len(set(row_lengths)) > 1:
        return False
    header = table[0]
    non_numeric = sum(1 for cell in header if cell and not cell.replace('.', '', 1).isdigit())
    return non_numeric >= 2

def extract_text_and_tables_with_fallback(page) -> Tuple[list, list]:
    text_blocks = []
    tables = []
    text = page.extract_text()
    if text:
        text_blocks.append(text.strip())

    for strategy in (
        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
        {"vertical_strategy": "text",  "horizontal_strategy": "text"}
    ):
        raw_tables = page.extract_tables(strategy)
        valid = []
        for table in raw_tables:
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
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)

    def process_page(i):
        with pdfplumber.open(pdf_path) as pdf_inner:
            return extract_text_and_tables_with_fallback(pdf_inner.pages[i])

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_page, range(num_pages)))

    all_text = []
    all_tables = []
    for text_blocks, tables in results:
        all_text.extend(text_blocks)
        all_tables.extend(tables)

    combined_text = "\n\n".join(all_text)
    # Ensure all tables have unique column names before concat
    for i in range(len(all_tables)):
        df = all_tables[i]
        if df.columns.duplicated().any():
            all_tables[i].columns = deduplicate_columns(df.columns)

    combined_df = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()


    return combined_text, combined_df

def build_rag_index_for_multiple(pdf_paths: List[str], api) -> RetrievalQA:
    docs = []
    for path in pdf_paths:
        text, _ = extract_all(path)
        filename = os.path.basename(path)
        docs.append(Document(page_content=text, metadata={"source": filename}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64}
    )
    vectorstore = InMemoryVectorStore.from_documents(chunks, embed_model)

    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=api  # Use env var for safety
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
