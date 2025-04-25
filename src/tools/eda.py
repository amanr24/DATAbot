'''
eda.py

Provides two tools for exploratory data analysis (EDA):
1. generate_sweetviz_report: creates an interactive Sweetviz HTML report.
2. explain_data: produces a narrative summary via get_dataframe_summary.

Caching is used to avoid re-computation on repeated calls in Streamlit.
'''

import os
import tempfile
from typing import Dict, Tuple

import streamlit as st  # For caching decorators
import pandas as pd
import sweetviz as sv  # Sweetviz for automated EDA reports

from tools.dataframe import get_dataframe_summary


def generate_sweetviz_report(
    data_raw: Dict[str, list],
    target: str = None,
    report_name: str = "Report.html",
    report_directory: str = None,
    open_browser: bool = False,
) -> Tuple[str, Dict]:
    '''
    Generate a Sweetviz HTML report for the first table in data_raw.

    Args:
        data_raw: Raw data as a dict of table_name -> list of records, or raw list.
        target: Optional target feature for specialized analysis.
        report_name: Filename for the HTML report.
        report_directory: Directory to write the report; a temp dir is created if None.
        open_browser: Whether to automatically open the report in a browser.

    Returns:
        A tuple of (message, artifact dict) where artifact contains 'report_path'.
    '''
    # Convert input to DataFrame (use first table if multiple)
    if isinstance(data_raw, dict):
        first_tbl = next(iter(data_raw.values()))
        df = pd.DataFrame(first_tbl)
    else:
        df = pd.DataFrame(data_raw)

    # Run Sweetviz analysis
    analysis = sv.analyze(df, target_feat=target)

    # Determine output directory
    if report_directory is None:
        report_directory = tempfile.mkdtemp()
    report_path = os.path.join(report_directory, report_name)

    # Generate HTML file
    analysis.show_html(filepath=report_path, open_browser=open_browser)

    # Return message and path for download link
    return (
        "Press on the download report button at the bottom",
        {"report_path": report_path},
    )


@st.cache_data(show_spinner=False)
def explain_data(
    data_raw,
    n_sample: int = 10
) -> str:
    '''
    Produce a narrative summary of the dataset using get_dataframe_summary.

    Args:
        data_raw: Raw data as dict of tables or raw list/dict for a single DataFrame.
        n_sample: Number of sample rows per column to include.

    Returns:
        Structured summary (list of dicts) as a string via get_dataframe_summary.
    '''
    print("    * Tool: explain_data")
    # Normalize input to dict of DataFrames
    if isinstance(data_raw, dict):
        df_dict: Dict[str, pd.DataFrame] = {}
        for name, table in data_raw.items():
            df_dict[name] = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
        summary = get_dataframe_summary(df_dict, n_sample, skip_stats=False)
    else:
        df = pd.DataFrame(data_raw)
        summary = get_dataframe_summary(df, n_sample, skip_stats=False)

    return summary
