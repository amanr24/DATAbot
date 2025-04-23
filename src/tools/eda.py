import streamlit as st
import pandas as pd
import sweetviz as sv
from tools.dataframe import get_dataframe_summary
from typing import Tuple, Dict
import os
import tempfile

# ✅ Cached narrative data summary tool
def generate_sweetviz_report(
    data_raw: Dict[str, list],
    target: str = None,
    report_name: str = "Report.html",
    report_directory: str = None,
    open_browser: bool = False,
) -> Tuple[str, Dict]:
    if isinstance(data_raw, dict):
        first_tbl = next(iter(data_raw.values()))
        df = pd.DataFrame(first_tbl)
    else:
        df = pd.DataFrame(data_raw)
    analysis = sv.analyze(df, target_feat=target)

    if report_directory is None:
        report_directory = tempfile.mkdtemp()
    report_path = os.path.join(report_directory, report_name)

    analysis.show_html(filepath=report_path, open_browser=open_browser)

    return (
        "Press on the download report button at the bottom",
        {"report_path": report_path},
    )

# ✅ Cache explain_data for performance on repeated EDA calls
@st.cache_data(show_spinner=False)
def explain_data(data_raw, n_sample=10) -> str:
    print("    * Tool: explain_data")
    if isinstance(data_raw, dict):
        df_dict = {}
        for name, table in data_raw.items():
            if isinstance(table, pd.DataFrame):
                df_dict[name] = table
            else:
                df_dict[name] = pd.DataFrame(table)
        summary = get_dataframe_summary(df_dict, n_sample, skip_stats=False)
    else:
        df = pd.DataFrame(data_raw)
        summary = get_dataframe_summary(df, n_sample, skip_stats=False)
    return summary
