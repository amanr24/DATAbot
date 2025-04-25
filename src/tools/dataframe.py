'''
dataframe.py

Defines get_dataframe_summary to produce structured summaries for DataFrames, lists, or dicts of DataFrames.
Includes metadata like shape, types, missing values, unique counts, and optional descriptive stats.
Caching via Streamlit to speed repeated calls.
'''

import pandas as pd
from typing import Union, List, Dict
import streamlit as st  # For caching decorator


@st.cache_data(show_spinner=False)
def get_dataframe_summary(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    n_sample: int = 10,
    skip_stats: bool = False,
) -> List[Dict]:
    '''
    Produce a summary for one or more DataFrames.

    Args:
        dataframes: Single DataFrame, list of DataFrames, or dict of named DataFrames.
        n_sample: Number of sample rows to include per column in the summary.
        skip_stats: If True, omit descriptive statistics section.

    Returns:
        List of dicts, each containing dataset metadata including:
          - dataset_name
          - shape (rows, columns)
          - column names and types
          - missing value percentages
          - unique value counts
          - optional describe stats (count, mean, median, min, max)

    Raises:
        TypeError: If input is not DataFrame, list, or dict of DataFrames.
    '''
    summaries = []

    # Handle dict of DataFrames
    if isinstance(dataframes, dict):
        for name, df in dataframes.items():
            summaries.append(_summarize_dataframe(df, name, skip_stats))
    # Handle single DataFrame
    elif isinstance(dataframes, pd.DataFrame):
        summaries.append(_summarize_dataframe(dataframes, "Single_Dataset", skip_stats))
    # Handle list of DataFrames
    elif isinstance(dataframes, list):
        for idx, df in enumerate(dataframes):
            dataset_name = f"Dataset_{idx}"
            summaries.append(_summarize_dataframe(df, dataset_name, skip_stats))
    else:
        raise TypeError("Input must be a DataFrame, list, or dictionary of DataFrames.")

    return summaries


def _summarize_dataframe(df: pd.DataFrame, dataset_name: str, skip_stats: bool) -> Dict:
    '''
    Internal helper to summarize a single DataFrame.

    Converts dict cells to strings, computes metadata, and optionally descriptive stats.

    Args:
        df: DataFrame to summarize.
        dataset_name: Identifier for this dataset.
        skip_stats: If True, skip numeric descriptive stats.

    Returns:
        Dict with summary information.
    '''
    # Work on a copy to avoid mutating original
    df_copy = df.copy()
    # Convert dict-like cells to strings
    df_copy = df_copy.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, dict) else x))

    # Identify numeric columns
    numeric_cols = df_copy.select_dtypes(include='number').columns.tolist()

    summary = {
        "dataset_name": dataset_name,
        "shape": {"rows": df_copy.shape[0], "columns": df_copy.shape[1]},
        "columns": df_copy.columns.tolist(),
        "column_types": df_copy.dtypes.astype(str).tolist(),
        "missing_percent": df_copy.isna().mean().round(4).mul(100).tolist(),
        "unique_value_counts": df_copy.nunique().tolist()
    }

    # Add descriptive stats if requested and numeric columns exist
    if not skip_stats and numeric_cols:
        desc = df_copy[numeric_cols].describe().loc[['count', 'mean', '50%', 'min', 'max']].round(2)
        summary["describe"] = {stat: desc.loc[stat].tolist() for stat in desc.index}

    return summary
