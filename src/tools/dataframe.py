import pandas as pd
from typing import Union, List, Dict
import streamlit as st

@st.cache_data(show_spinner=False)
def get_dataframe_summary(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    n_sample: int = 10,
    skip_stats: bool = False,
) -> List[Dict]:
    summaries = []

    if isinstance(dataframes, dict):
        for dataset_name, df in dataframes.items():
            summaries.append(_summarize_dataframe(df, dataset_name, skip_stats))
    elif isinstance(dataframes, pd.DataFrame):
        summaries.append(_summarize_dataframe(dataframes, "Single_Dataset", skip_stats))
    elif isinstance(dataframes, list):
        for idx, df in enumerate(dataframes):
            dataset_name = f"Dataset_{idx}"
            summaries.append(_summarize_dataframe(df, dataset_name, skip_stats))
    else:
        raise TypeError("Input must be a DataFrame, list, or dictionary of DataFrames.")

    return summaries

def _summarize_dataframe(df: pd.DataFrame, dataset_name: str, skip_stats: bool) -> Dict:
    df = df.copy()
    df = df.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, dict) else x))
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    column_types = df.dtypes.astype(str).tolist()
    columns = df.columns.tolist()

    summary = {
        "dataset_name": dataset_name,
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": columns,
        "column_types": column_types,
        "missing_percent": df.isna().mean().round(4).mul(100).tolist(),
        "unique_value_counts": df.nunique().tolist()
    }

    if not skip_stats and numeric_cols:
        desc = df[numeric_cols].describe().loc[['count', 'mean', '50%', 'min', 'max']].round(2)
        summary["describe"] = {
            stat: desc.loc[stat].tolist()
            for stat in desc.index
        }

    return summary

