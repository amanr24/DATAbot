import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pickle
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from pycaret.classification import setup as cls_setup, compare_models as cls_compare, save_model as cls_save, predict_model as predict_model_cls, pull as cls_pull, get_leaderboard as cls_leaderboard
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, save_model as reg_save, predict_model as predict_model_reg, pull as reg_pull, get_leaderboard as reg_leaderboard
except ImportError:
    predict_model_cls = None
    predict_model_reg = None

# Helper for ML-based imputation of target column
def ml_impute_target(df, target):
    from pycaret.regression import setup, compare_models, predict_model
    train_df = df[df[target].notnull()]
    test_df = df[df[target].isnull()]
    if len(test_df) == 0:
        return df
    feature_cols = [col for col in df.columns if col != target]
    s = setup(data=train_df, target=target, session_id=123, fold=3, train_size=0.8, normalize=True)
    best_model = compare_models()
    preds = predict_model(best_model, data=test_df[feature_cols])
    df.loc[df[target].isnull(), target] = preds['Label']
    return df

def render_dashboard(df: pd.DataFrame):
    if df is None or df.empty or len(df.columns) == 0:
        st.warning("Please upload a valid CSV file with data before proceeding.")
        st.stop()
    st.header("📊 DataBot Dashboard")
    tabs = st.tabs(["Data Cleaning & AutoML", "What-If Simulator", "Forecast Playground"])

    # --- Data Cleaning & AutoML Tab ---
    with tabs[0]:
        st.subheader("Data Cleaning, EDA & AutoML")
        if 'cleaned_df' in st.session_state:
            cleaned_df = st.session_state['cleaned_df']
        else:
            cleaned_df = df.copy()
        # --- Data Cleaning ---
        st.markdown("### Data Cleaning")
        cleaning_method = st.selectbox(
            "Select Data Cleaning Method for Missing Values",
            ["Auto (Recommended)", "Drop Rows with Missing Values", "Fill with Mean", "Fill with Median", "Fill with Mode", "Custom Value for Each Column"]
        )
        cleaned_df = cleaned_df.copy()
        if cleaning_method == "Auto (Recommended)":
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'O':
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
                    if cleaned_df[col].dtype == 'O':
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if cleaned_df[col].mode().size > 0 else "Unknown")
                elif pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                else:
                    cleaned_df[col] = cleaned_df[col].fillna("Unknown")
            st.success("Auto-cleaned: Numeric columns filled with median, categoricals with mode.")
        elif cleaning_method == "Drop Rows with Missing Values":
            cleaned_df = cleaned_df.dropna()
            st.success("Dropped all rows with missing values.")
        elif cleaning_method == "Fill with Mean":
            for col in cleaned_df.select_dtypes(include=['number']).columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            for col in cleaned_df.select_dtypes(include=['object', 'category']).columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if cleaned_df[col].mode().size > 0 else "Unknown")
            st.success("Filled numeric columns with mean, categoricals with mode.")
        elif cleaning_method == "Fill with Median":
            for col in cleaned_df.select_dtypes(include=['number']).columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            for col in cleaned_df.select_dtypes(include=['object', 'category']).columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if cleaned_df[col].mode().size > 0 else "Unknown")
            st.success("Filled numeric columns with median, categoricals with mode.")
        elif cleaning_method == "Fill with Mode":
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if cleaned_df[col].mode().size > 0 else "Unknown")
            st.success("Filled all columns with mode.")
        elif cleaning_method == "Custom Value for Each Column":
            custom_values = {}
            for col in cleaned_df.columns:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    val = st.text_input(f"Custom value for {col} (numeric)", value="", key=f"custom_{col}")
                    if val != "":
                        try:
                            val = float(val)
                            custom_values[col] = val
                        except Exception:
                            st.warning(f"Invalid numeric value for {col}, skipping.")
                else:
                    val = st.text_input(f"Custom value for {col} (categorical)", value="", key=f"custom_{col}")
                    if val != "":
                        custom_values[col] = val
            if st.button("Apply Custom Values"):
                for col, val in custom_values.items():
                    cleaned_df[col] = cleaned_df[col].fillna(val)
                st.success("Filled missing values with custom values where provided.")
        # Convert datetime columns to string to avoid serialization errors
        for col in cleaned_df.columns:
            if np.issubdtype(cleaned_df[col].dtype, np.datetime64):
                cleaned_df[col] = cleaned_df[col].astype(str)
        # Encode all categorical columns
        for col in cleaned_df.select_dtypes(include=['object', 'category']).columns:
            cleaned_df[col] = cleaned_df[col].astype('category').cat.codes
        st.session_state['cleaned_df'] = cleaned_df
        st.write("Shape:", cleaned_df.shape)
        st.write("Summary Statistics:")
        st.dataframe(cleaned_df.describe(include='all'))
        st.write("Column Types:")
        st.write(cleaned_df.dtypes)
        st.write("Correlation Matrix:")
        st.dataframe(cleaned_df.corr(numeric_only=True))
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            st.write("Distributions:")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(cleaned_df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
        if st.button("Show Pairplot (numeric columns)"):
            fig = sns.pairplot(cleaned_df[numeric_cols].dropna())
            st.pyplot(fig)
        st.markdown("---")
        # Show cleaned data preview and missing values summary
        st.markdown("#### Cleaned Data Preview (used for AutoML)")
        st.dataframe(cleaned_df.head(30))
        st.markdown(f"**Total Rows:** {cleaned_df.shape[0]}, **Total Columns:** {cleaned_df.shape[1]}")
        missing_per_col = cleaned_df.isnull().sum()
        total_missing_rows = (cleaned_df.isnull().sum(axis=1) > 0).sum()
        st.markdown(f"**Rows with any missing values:** {total_missing_rows}")
        st.markdown("**Missing values per column:")
        st.dataframe(missing_per_col)
        # Data Quality Metrics
        st.markdown("**Data Quality Metrics:**")
        def get_quality_metrics(df):
            metrics = {}
            for col in df.columns:
                data = df[col]
                missing = data.isnull().mean() * 100
                n_unique = data.nunique(dropna=True)
                if pd.api.types.is_numeric_dtype(data):
                    skew = data.dropna().skew()
                    q1 = data.dropna().quantile(0.25)
                    q3 = data.dropna().quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((data < (q1 - 1.5*iqr)) | (data > (q3 + 1.5*iqr))).sum() / data.dropna().shape[0] * 100 if data.dropna().shape[0] > 0 else 0
                else:
                    skew = 0
                    outliers = 0
                metrics[col] = {
                    'Missing %': missing,
                    'Cardinality': n_unique,
                    'Skewness': abs(skew),
                    'Outliers %': outliers
                }
            return pd.DataFrame(metrics).T
        st.dataframe(get_quality_metrics(cleaned_df))
        st.markdown("---")
        # Download cleaned dataset
        st.download_button(
            "Download Cleaned Dataset as CSV",
            cleaned_df.to_csv(index=False),
            "cleaned_data.csv",
            key="eda_cleaned_download"
        )
        st.markdown("---")
        # --- AutoML ---
        st.markdown("### AutoML (PyCaret)")
        st.info("AutoML will use the cleaned dataset above.")
        fast_mode = st.checkbox("Fast Mode (fewer models, faster results)", value=True)
        target_col = st.selectbox("Select Target Column for AutoML", cleaned_df.columns)
        feature_cols = st.multiselect(
            "Select Features (leave blank for all except target)",
            [col for col in cleaned_df.columns if col != target_col],
            help="Choose which features to use for modeling."
        )
        test_size = st.slider("Test Size (percent for test set)", min_value=10, max_value=50, value=20, step=5)
        folds = st.number_input("Cross-Validation Folds", min_value=2, max_value=10, value=3 if fast_mode else 5)
        st.write(f"Features: {feature_cols if feature_cols else 'All except target'} | Test Size: {test_size}% | Folds: {folds}")
        if st.button("Run AutoML"):
            status_placeholder = st.empty()
            leaderboard_df = None
            results_df = None
            best_model = None
            try:
                status_placeholder.info("Step 1: Preparing data...")
                # Prepare data
                X = cleaned_df.copy()
                if feature_cols:
                    X = X[feature_cols + [target_col]]
                y = X[target_col]
                # Save feature columns and dtypes for What-If tab
                X.to_pickle("X_metadata.pkl")
                y.to_pickle("y_metadata.pkl")
                from pycaret.classification import setup as cls_setup, compare_models as cls_compare, save_model as cls_save, pull as cls_pull, get_leaderboard as cls_leaderboard
                from pycaret.regression import setup as reg_setup, compare_models as reg_compare, save_model as reg_save, predict_model as predict_model_reg, pull as reg_pull, get_leaderboard as reg_leaderboard
                compare_kwargs = {}
                # Dynamically select available estimators for fast mode
                if fast_mode:
                    if y.nunique() <= 20 and y.dtype in [int, float]:
                        # Classification
                        try:
                            from pycaret.classification import models as cls_models
                            available = [m['id'] for m in cls_models()]
                            fast_models = [m for m in ['lr', 'dt', 'knn', 'ridge', 'lasso', 'rf'] if m in available]
                            if fast_models:
                                compare_kwargs['include'] = fast_models
                        except Exception:
                            compare_kwargs = {}
                    else:
                        # Regression
                        try:
                            from pycaret.regression import models as reg_models
                            available = [m['id'] for m in reg_models()]
                            fast_models = [m for m in ['lr', 'dt', 'knn', 'ridge', 'lasso', 'rf'] if m in available]
                            if fast_models:
                                compare_kwargs['include'] = fast_models
                        except Exception:
                            compare_kwargs = {}
                if y.nunique() <= 20 and y.dtype in [int, float]:
                    status_placeholder.info("Step 2: Setting up classification...")
                    s = cls_setup(X, target=target_col, session_id=123, fold=folds, train_size=1-test_size/100, normalize=True)
                    status_placeholder.info("Step 3: Comparing classification models...")
                    try:
                        best_model = cls_compare(**compare_kwargs)
                        # Ensure best_model is a single model, not a list/array
                        if isinstance(best_model, (list, np.ndarray)):
                            best_model = best_model[0]
                    except Exception:
                        status_placeholder.warning("Some estimators not available, running with default models.")
                        best_model = cls_compare()
                        if isinstance(best_model, (list, np.ndarray)):
                            best_model = best_model[0]
                    leaderboard_df = cls_leaderboard()
                    results_df = cls_pull()
                    status_placeholder.info("Step 4: Saving best classification model...")
                    cls_save(best_model, 'best_model')
                    status_placeholder.success("Classification model trained and saved as best_model.pkl!")
                else:
                    status_placeholder.info("Step 2: Setting up regression...")
                    s = reg_setup(X, target=target_col, session_id=123, fold=folds, train_size=1-test_size/100, normalize=True)
                    status_placeholder.info("Step 3: Comparing regression models...")
                    try:
                        best_model = reg_compare(**compare_kwargs)
                        if isinstance(best_model, (list, np.ndarray)):
                            best_model = best_model[0]
                    except Exception:
                        status_placeholder.warning("Some estimators not available, running with default models.")
                        best_model = reg_compare()
                        if isinstance(best_model, (list, np.ndarray)):
                            best_model = best_model[0]
                    leaderboard_df = reg_leaderboard()
                    results_df = reg_pull()
                    status_placeholder.info("Step 4: Saving best regression model...")
                    reg_save(best_model, 'best_model')
                    status_placeholder.success("Regression model trained and saved as best_model.pkl!")
                # Show results
                if leaderboard_df is not None:
                    leaderboard_df = leaderboard_df.astype(str)
                    leaderboard_df = leaderboard_df[~leaderboard_df.iloc[:,0].isin(["Mean", "Std", "mean", "std"])]
                    st.markdown("#### AutoML Leaderboard")
                    st.dataframe(leaderboard_df)
                    st.download_button("Download Leaderboard as CSV", leaderboard_df.to_csv(index=False), "leaderboard.csv")
                if results_df is not None:
                    results_df = results_df.astype(str)
                    results_df = results_df[~results_df.iloc[:,0].isin(["Mean", "Std", "mean", "std"])]
                    st.markdown("#### Model Results")
                    st.dataframe(results_df)
                if best_model is not None:
                    st.markdown("#### Best Model Summary")
                    st.write(str(best_model))
            except Exception as e:
                status_placeholder.error(f"AutoML failed: {e}\nTrying to auto-fix and re-run...")
                for col in cleaned_df.columns:
                    if cleaned_df[col].dtype == 'O':
                        cleaned_df[col] = cleaned_df[col].astype('category').cat.codes
                try:
                    status_placeholder.info("Retry: Preparing data after auto-fix...")
                    X = cleaned_df.copy()
                    if feature_cols:
                        X = X[feature_cols + [target_col]]
                    y = X[target_col]
                    # Save feature columns and dtypes for What-If tab
                    X.to_pickle("X_metadata.pkl")
                    y.to_pickle("y_metadata.pkl")
                    if fast_mode:
                        if y.nunique() <= 20 and y.dtype in [int, float]:
                            try:
                                from pycaret.classification import models as cls_models
                                available = [m['id'] for m in cls_models()]
                                fast_models = [m for m in ['lr', 'dt', 'knn', 'ridge', 'lasso', 'rf'] if m in available]
                                if fast_models:
                                    compare_kwargs['include'] = fast_models
                            except Exception:
                                compare_kwargs = {}
                        else:
                            try:
                                from pycaret.regression import models as reg_models
                                available = [m['id'] for m in reg_models()]
                                fast_models = [m for m in ['lr', 'dt', 'knn', 'ridge', 'lasso', 'rf'] if m in available]
                                if fast_models:
                                    compare_kwargs['include'] = fast_models
                            except Exception:
                                compare_kwargs = {}
                    if y.nunique() <= 20 and y.dtype in [int, float]:
                        status_placeholder.info("Retry: Setting up classification...")
                        s = cls_setup(X, target=target_col, session_id=123, fold=folds, train_size=1-test_size/100, normalize=True)
                        status_placeholder.info("Retry: Comparing classification models...")
                        try:
                            best_model = cls_compare(**compare_kwargs)
                            if isinstance(best_model, (list, np.ndarray)):
                                best_model = best_model[0]
                        except Exception:
                            status_placeholder.warning("Some estimators not available, running with default models.")
                            best_model = cls_compare()
                            if isinstance(best_model, (list, np.ndarray)):
                                best_model = best_model[0]
                        leaderboard_df = cls_leaderboard()
                        results_df = cls_pull()
                        status_placeholder.info("Retry: Saving best classification model...")
                        cls_save(best_model, 'best_model')
                        status_placeholder.success("Classification model trained and saved as best_model.pkl!")
                    else:
                        status_placeholder.info("Retry: Setting up regression...")
                        s = reg_setup(X, target=target_col, session_id=123, fold=folds, train_size=1-test_size/100, normalize=True)
                        status_placeholder.info("Retry: Comparing regression models...")
                        try:
                            best_model = reg_compare(**compare_kwargs)
                            if isinstance(best_model, (list, np.ndarray)):
                                best_model = best_model[0]
                        except Exception:
                            status_placeholder.warning("Some estimators not available, running with default models.")
                            best_model = reg_compare()
                            if isinstance(best_model, (list, np.ndarray)):
                                best_model = best_model[0]
                        leaderboard_df = reg_leaderboard()
                        results_df = reg_pull()
                        status_placeholder.info("Retry: Saving best regression model...")
                        reg_save(best_model, 'best_model')
                        status_placeholder.success("Regression model trained and saved as best_model.pkl!")
                    # Show results
                    if leaderboard_df is not None:
                        leaderboard_df = leaderboard_df.astype(str)
                        leaderboard_df = leaderboard_df[~leaderboard_df.iloc[:,0].isin(["Mean", "Std", "mean", "std"])]
                        st.markdown("#### AutoML Leaderboard")
                        st.dataframe(leaderboard_df)
                        st.download_button("Download Leaderboard as CSV", leaderboard_df.to_csv(index=False), "leaderboard.csv")
                    if results_df is not None:
                        results_df = results_df.astype(str)
                        results_df = results_df[~results_df.iloc[:,0].isin(["Mean", "Std", "mean", "std"])]
                        st.markdown("#### Model Results")
                        st.dataframe(results_df)
                    if best_model is not None:
                        st.markdown("#### Best Model Summary")
                        st.write(str(best_model))
                except Exception as e2:
                    status_placeholder.error(f"AutoML failed again: {e2}")
        st.markdown("---")
        # EDA Report
        if st.button("Generate EDA Report"):
            with st.spinner("Generating EDA report..."):
                profile = ProfileReport(cleaned_df, minimal=True)
                profile.to_file("eda_report.html")
                with open("eda_report.html", "r", encoding="utf-8") as f:
                    html = f.read()
                st.success("EDA report generated!")
                st.download_button("Download EDA Report (HTML)", html, "eda_report.html")
                components.html(html, height=600, scrolling=True)
        st.markdown("---")

    # What-If Scenario Simulator Tab
    with tabs[1]:
        st.header("What-If Scenario Simulator")
        st.info("Adjust features and predict using the trained model.")
        import os, pickle
        model_path = "best_model.pkl"
        X_meta_path = "X_metadata.pkl"
        y_meta_path = "y_metadata.pkl"
        best_model = None
        X = None
        y = None
        # Try to load model and metadata if not in memory
        if best_model is None or X is None or y is None:
            if os.path.exists(model_path) and os.path.exists(X_meta_path) and os.path.exists(y_meta_path):
                try:
                    if y is not None and y.nunique() <= 20 and y.dtype in [int, float]:
                        from pycaret.classification import load_model as cls_load
                        best_model_loaded = cls_load('best_model')
                    else:
                        from pycaret.regression import load_model as reg_load
                        best_model_loaded = reg_load('best_model')
                    # Ensure we get the model object, not a string
                    if isinstance(best_model_loaded, str):
                        st.error("Model loading failed: got a string instead of model object. Please retrain your model.")
                        best_model = None
                    else:
                        best_model = best_model_loaded
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    best_model = None
                X = pd.read_pickle(X_meta_path)
                y = pd.read_pickle(y_meta_path)
        if best_model is not None and X is not None and y is not None:
            # Only use features that the model actually expects
            try:
                # Try to get feature names from the model pipeline (works for most PyCaret/sklearn models)
                if hasattr(best_model, 'feature_names_in_'):
                    model_features = list(best_model.feature_names_in_)
                elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps, 'final_estimator_') and hasattr(best_model.named_steps.final_estimator_, 'feature_names_in_'):
                    model_features = list(best_model.named_steps.final_estimator_.feature_names_in_)
                else:
                    model_features = list(X.columns)
            except Exception:
                model_features = list(X.columns)
            expected_cols = model_features
            user_inputs = {}
            for col in expected_cols:
                dtype = X[col].dtype if col in X.columns else float
                if pd.api.types.is_numeric_dtype(dtype):
                    user_inputs[col] = st.number_input(f"{col}", value=float(X[col].median()) if col in X.columns and not X[col].isnull().all() else 0.0)
                else:
                    options = list(X[col].dropna().unique()) if col in X.columns else ["Unknown"]
                    if not options:
                        options = ["Unknown"]
                    user_inputs[col] = st.selectbox(f"{col}", options)
            if st.button("Predict What-If Scenario"):
                try:
                    scenario_df = pd.DataFrame([user_inputs])
                    # Align columns and dtypes
                    for col in expected_cols:
                        if col not in scenario_df.columns:
                            scenario_df[col] = np.nan
                        # Convert types to match training
                        if col in X.columns:
                            scenario_df[col] = scenario_df[col].astype(X[col].dtype)
                    scenario_df = scenario_df[expected_cols]
                    # Ensure best_model is a single model, not a list/array
                    if isinstance(best_model, (list, np.ndarray)):
                        best_model = best_model[0]
                    # Predict
                    from pycaret.classification import predict_model as cls_predict
                    from pycaret.regression import predict_model as reg_predict
                    if y.nunique() <= 20 and y.dtype in [int, float]:
                        pred = cls_predict(best_model, data=scenario_df)
                    else:
                        pred = reg_predict(best_model, data=scenario_df)
                    prediction = pred.iloc[0,-1]
                    st.success(f"Prediction: {prediction}")
                    # Summarize the result
                    st.markdown("#### Prediction Summary")
                    st.write(f"Based on your input scenario, the model predicts: **{prediction}**.")
                    st.write("---")
                    st.markdown("**Input scenario details:**")
                    st.dataframe(scenario_df)
                    if y.nunique() <= 20 and y.dtype in [int, float] and 'Score' in pred.columns:
                        st.write(f"Probability Score: {pred['Score'].iloc[0]:.2f}")
                except Exception as e:
                    st.error(f"Prediction failed. Please check your inputs. Error: {e}")
        else:
            st.warning("Train a model first using AutoML.")

    # Forecast Playground Tab
    with tabs[2]:
        st.subheader("📈 AI-Powered Forecast Playground")
        st.markdown("Select a time series column and generate interactive forecasts.")
        if not PROPHET_AVAILABLE:
            st.warning("Prophet library is not installed. Please install prophet to use forecasting features.")
        else:
            if 'cleaned_df' in st.session_state:
                cleaned_df = st.session_state['cleaned_df']
            date_cols = [col for col in cleaned_df.columns if pd.api.types.is_datetime64_any_dtype(cleaned_df[col])]
            if not date_cols:
                for col in cleaned_df.columns:
                    try:
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                        date_cols.append(col)
                        break
                    except Exception:
                        continue
            if date_cols:
                ds_col = st.selectbox("Select Date Column", date_cols)
                y_col = st.selectbox("Select Value Column to Forecast", [col for col in cleaned_df.columns if col != ds_col and pd.api.types.is_numeric_dtype(cleaned_df[col])])
                periods = st.number_input("Forecast Horizon (days)", min_value=1, max_value=365, value=30)
                if st.button("Run Forecast"):
                    ts_df = cleaned_df[[ds_col, y_col]].dropna().rename(columns={ds_col: "ds", y_col: "y"})
                    m = Prophet()
                    m.fit(ts_df)
                    future = m.make_future_dataframe(periods=int(periods))
                    forecast = m.predict(future)
                    fig1 = m.plot(forecast)
                    st.pyplot(fig1)
                    fig2 = m.plot_components(forecast)
                    st.pyplot(fig2)
                    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(20))
            else:
                st.warning("No date/time column found in your data.")

if __name__ == "__main__":
    render_dashboard(pd.DataFrame())
