'''
sql.py

Utility functions for extracting database metadata and sampling data, with safe identifier quoting 
and dialect-aware row limiting, to support downstream SQL agents.
'''

import pandas as pd
import sqlalchemy as sql
from sqlalchemy import inspect


def get_db_metadata(connection, n_samples=3) -> dict:
    '''
    Collects metadata and sample values from a SQL database.

    - Safely quotes identifiers for different dialects.
    - Retrieves a few sample values per column to aid schema understanding.

    Parameters:
        connection: SQLAlchemy Connection or Engine instance.
        n_samples: Number of sample rows to retrieve for each column.

    Returns:
        metadata dict with keys:
          - dialect, driver, connection_url
          - schemas: list of schema objects, each with table and column info.
    '''
    # Determine if passed object is Engine or Connection
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection

    # Initialize metadata structure
    metadata = {
        "dialect": None,
        "driver": None,
        "connection_url": None,
        "schemas": [],
    }

    try:
        # Extract basic connection info
        engine = conn.engine
        metadata["dialect"] = engine.dialect.name
        metadata["driver"] = engine.driver
        metadata["connection_url"] = str(engine.url)

        # Use inspector to traverse schemas and tables
        inspector = inspect(engine)
        preparer = inspector.bind.dialect.identifier_preparer

        # Loop over each schema (e.g., 'main' for SQLite)
        for schema_name in inspector.get_schema_names():
            schema_obj = {"schema_name": schema_name, "tables": []}
            table_names = inspector.get_table_names(schema=schema_name)

            for table_name in table_names:
                table_info = {
                    "table_name": table_name,
                    "columns": [],
                    "primary_key": [],
                    "foreign_keys": [],
                    "indexes": []
                }

                # Iterate over columns to sample values
                columns = inspector.get_columns(table_name, schema=schema_name)
                for col in columns:
                    col_name = col["name"]
                    col_type = str(col["type"])

                    # Fully-quote schema.table and column identifiers
                    table_q = f"{preparer.quote_identifier(schema_name)}.{preparer.quote_identifier(table_name)}"
                    col_q = preparer.quote_identifier(col_name)
                    # Build sample query depending on dialect
                    sample_query = build_query(col_q, table_q, n_samples, metadata["dialect"])

                    try:
                        df_sample = pd.read_sql(sample_query, conn)
                        samples = df_sample[col_name].head(n_samples).tolist()
                    except Exception as err:
                        samples = [f"Error retrieving data: {err}"]

                    # Append column metadata
                    table_info["columns"].append({
                        "name": col_name,
                        "type": col_type,
                        "sample_values": samples
                    })

                # Primary key constraint info
                pk = inspector.get_pk_constraint(table_name, schema=schema_name)
                table_info["primary_key"] = pk.get("constrained_columns", [])

                # Foreign key relationships
                fks = inspector.get_foreign_keys(table_name, schema=schema_name)
                table_info["foreign_keys"] = [
                    {"local_cols": fk["constrained_columns"],
                     "referred_table": fk["referred_table"],
                     "referred_cols": fk["referred_columns"]}
                    for fk in fks
                ]

                # Index metadata
                idxs = inspector.get_indexes(table_name, schema=schema_name)
                table_info["indexes"] = idxs

                schema_obj["tables"].append(table_info)

            metadata["schemas"].append(schema_obj)

    finally:
        # Close Engine connection if opened within
        if is_engine:
            conn.close()

    return metadata


def build_query(col_name_quoted: str, table_name_quoted: str, n: int, dialect_name: str) -> str:
    '''
    Construct a SQL query to fetch a limited number of rows per column.

    - For SQLite: uses ORDER BY RANDOM() LIMIT n
    - For Oracle-like dialects: uses WHERE ROWNUM <= n

    Parameters:
        col_name_quoted: properly quoted column identifier
        table_name_quoted: properly quoted table identifier
        n: number of rows to fetch
        dialect_name: database dialect name string

    Returns:
        SQL query string
    '''
    if "sqlite" in dialect_name:
        return f"SELECT {col_name_quoted} FROM {table_name_quoted} ORDER BY RANDOM() LIMIT {n}"
    # Generic fallback for ROWNUM-style limits
    return f"SELECT {col_name_quoted} FROM {table_name_quoted} WHERE ROWNUM <= {n}"
