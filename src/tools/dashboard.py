# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pickle

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from pycaret.classification import predict_model as predict_model_cls
    from pycaret.regression import predict_model as predict_model_reg
except ImportError:
    predict_model_cls = None
    predict_model_reg = None

# Helper for ML-based imputation of target column# ✅ Cached ML-based imputation of target column
@st.cache_data(show_spinner=True)
def ml_impute_target(df, target):
    from pycaret.regression import setup, compare_models, predict_model
    train_df = df[df[target].notnull()]
    test_df = df[df[target].isnull()]
    if len(test_df) == 0:
        return df
    feature_cols = [col for col in df.columns if col != target]
    s = setup(data=train_df, target=target, silent=True, session_id=123, fold=3, train_size=0.8, normalize=True)
    best_model = compare_models()
    preds = predict_model(best_model, data=test_df[feature_cols])
    df.loc[df[target].isnull(), target] = preds['Label']
    return df

def render_dashboard(df: pd.DataFrame):
    if df is None or df.empty or len(df.columns) == 0:
        st.warning("Please upload a valid CSV file with data before proceeding.")
        st.stop()
    st.header("📊 DataBot Dashboard")
    tabs = st.tabs(["EDA", "AutoML", "What-If Simulator", "Forecast Playground"])

    # --- Data Cleaning is now part of AutoML tab ---

    # EDA Tab
    with tabs[0]:
        st.subheader("Exploratory Data Analysis (EDA)")
        st.markdown("You can clean your data in the AutoML tab. EDA always shows the latest cleaned dataset.")
        if 'cleaned_df' in st.session_state:
            cleaned_df = st.session_state['cleaned_df']
        else:
            cleaned_df = df.copy()
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
        st.download_button(
            "Download Cleaned Dataset as CSV",
            cleaned_df.to_csv(index=False),
            "cleaned_data.csv",
            key="eda_cleaned_download"
        )
    
    # AutoML Tab (includes Data Cleaning)
    with tabs[1]:
        st.subheader("AutoML (PyCaret) + Data Cleaning")
        st.markdown("Clean your data, then run AutoML. All downstream features use the cleaned dataset.")
        cleaned_df = df.copy()
        cleaning_action = st.radio(
            "Select Data Cleaning Action",
            ("None", "Drop Rows with Missing Values", "Fill Missing Values", "AutoML Fill Target"),
            horizontal=True
        )
        if cleaning_action == "Drop Rows with Missing Values":
            cleaned_df = cleaned_df.dropna()
            st.success("Dropped rows with missing values.")
        elif cleaning_action == "Fill Missing Values":
            impute_method = st.selectbox(
                "Imputation Method",
                ("Auto", "Mean", "Median", "Mode", "Custom Value"),
                help="Choose how to fill missing values. 'Auto' uses mean/median for numeric and mode for categorical columns."
            )
            custom_value = None
            if impute_method == "Custom Value":
                col_types = cleaned_df.dtypes
                for col in cleaned_df.columns:
                    if cleaned_df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            val = st.text_input(f"Enter numeric value for {col}", key=f"custom_{col}")
                            try:
                                val = float(val) if val != '' else None
                            except Exception:
                                val = None
                            if val is not None:
                                cleaned_df[col] = cleaned_df[col].fillna(val)
                            elif val == '':
                                pass
                            else:
                                st.warning(f"Please enter a valid numeric value for {col}.")
                        else:
                            val = st.text_input(f"Enter string value for {col}", key=f"custom_{col}")
                            if val:
                                cleaned_df[col] = cleaned_df[col].fillna(val)
                if st.button("Apply Imputation"):
                    st.success("Filled missing values with custom values (validated by type).")
            else:
                if st.button("Apply Imputation"):
                    if impute_method == "Auto":
                        for col in cleaned_df.columns:
                            if cleaned_df[col].isnull().any():
                                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                                else:
                                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0])
                        st.success("Auto-imputation applied: mean/median for numeric, mode for categorical.")
                    elif impute_method == "Mean":
                        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                        st.success("Filled numeric columns with mean.")
                    elif impute_method == "Median":
                        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                        st.success("Filled numeric columns with median.")
                    elif impute_method == "Mode":
                        for col in cleaned_df.columns:
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0])
                        st.success("Filled all columns with mode.")
        elif cleaning_action == "AutoML Fill Target":
            target_col = st.selectbox("Select Target Column to Impute with AutoML", options=[col for col in cleaned_df.columns if cleaned_df[col].isnull().sum() > 0], help="Only columns with missing values are shown.")
            if st.button("Run AutoML Imputation"):
                with st.spinner("Running AutoML to fill missing values in target column..."):
                    cleaned_df = ml_impute_target(cleaned_df, target_col)
                st.success(f"Filled missing values in '{target_col}' using AutoML regression.")
        else:
            st.info("No cleaning action applied.")
        
        st.session_state['cleaned_df'] = cleaned_df
        st.write("Preview Cleaned Data:")
        st.dataframe(cleaned_df.head())
        st.markdown("---")
        st.download_button(
            "Download Cleaned Dataset as CSV",
            cleaned_df.to_csv(index=False),
            "cleaned_data.csv",
            key="automl_cleaned_download"
        )

        st.markdown("---")
        st.subheader("AutoML")

        # --- AutoML Target and Task Suggestion ---
        # Analyze columns for best regression/classification target
        classification_candidates = []
        regression_candidates = []
        for col in cleaned_df.columns:
            nunique = cleaned_df[col].nunique(dropna=True)
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                if nunique <= 20 and nunique > 1:
                    classification_candidates.append((col, nunique, 'numeric'))
                elif nunique > 20:
                    regression_candidates.append((col, nunique, 'numeric'))
            elif pd.api.types.is_object_dtype(cleaned_df[col]) or pd.api.types.is_categorical_dtype(cleaned_df[col]):
                if nunique <= 20 and nunique > 1:
                    classification_candidates.append((col, nunique, 'categorical'))
        suggested_task = None
        suggested_target = None
        if classification_candidates:
            suggested_task = 'Classification'
            suggested_target = classification_candidates[0][0]
        elif regression_candidates:
            suggested_task = 'Regression'
            suggested_target = regression_candidates[0][0]
        st.markdown("**AutoML Suggestions:**")
        if suggested_task and suggested_target:
            st.info(f"Suggested Task: {suggested_task} | Suggested Target: {suggested_target}")
        else:
            st.warning("No clear target column found for classification or regression. Please select manually.")

        module = st.selectbox(
            "Select AutoML Task Type",
            ["Classification", "Regression", "Clustering"],
            index=["Classification", "Regression", "Clustering"].index(suggested_task) if suggested_task else 0,
            help="Choose the type of ML problem you want to solve."
        )
        target = None
        if module != "Clustering":
            default_target = suggested_target if suggested_target and suggested_task == module else None
            target = st.selectbox("Select Target Column", options=cleaned_df.columns, index=list(cleaned_df.columns).index(default_target) if default_target else 0, help="This is the column you want to predict.")
        features = st.multiselect(
            "Select Features to Include (leave blank for all)",
            options=[col for col in cleaned_df.columns if col != target],
            help="You can restrict AutoML to specific columns if you want."
        )
        if features:
            X = cleaned_df[features]
            if target:
                X[target] = cleaned_df[target]
            data = X
        else:
            data = cleaned_df
        if module != "Clustering" and target is not None:
            if data[target].isnull().any():
                st.warning(f"{data[target].isnull().sum()} missing values found in the target column '{target}'. These rows will be removed automatically for AutoML.")
                data = data[data[target].notnull()]
        st.markdown("**Preprocessing Options**")
        normalize = st.checkbox("Normalize/Scale Numeric Features", value=True)
        split = st.slider("Train/Test Split %", min_value=50, max_value=90, value=80, step=5)
        folds = st.number_input("Cross-Validation Folds", min_value=2, max_value=10, value=5)
        st.markdown("**Advanced Options**")
        custom_metric = st.text_input("Custom Metric (optional)", help="e.g. 'AUC', 'F1', 'RMSE'")
        if st.button("Run AutoML"):
            with st.spinner("Running AutoML. Please wait..."):
                try:
                    if module == "Classification":
                        from pycaret.classification import setup, compare_models, pull, get_leaderboard, plot_model, save_model, load_model
                    elif module == "Regression":
                        from pycaret.regression import setup, compare_models, pull, get_leaderboard, plot_model, save_model, load_model
                    else:
                        from pycaret.clustering import setup, create_model, pull, get_leaderboard, plot_model, save_model, load_model
                    setup_kwargs = dict(
                        data=data,
                        session_id=123,
                        fold=folds,
                        train_size=split/100,
                        normalize=normalize
                    )
                    if module != "Clustering":
                        setup_kwargs["target"] = target
                    s = setup(**setup_kwargs)
                    if module == "Clustering":
                        best_model = create_model("kmeans")
                    else:
                        if custom_metric:
                            try:
                                best_model = compare_models(sort=custom_metric)
                            except Exception:
                                st.warning("Custom metric not supported, using default sort.")
                                best_model = compare_models()
                        else:
                            best_model = compare_models()
                    leaderboard = get_leaderboard()
                    st.write("Leaderboard:")
                    st.dataframe(leaderboard)
                    st.write("Best Model:", best_model)
                    st.write("Results:")
                    st.dataframe(pull())
                    with open("best_model.pkl", "wb") as f:
                        pickle.dump(best_model, f)
                    st.success("Best model saved for What-If Scenario Simulator!")
                    st.markdown("**Model Plots**")
                    if module != "Clustering":
                        for plot_type in ["auc", "confusion_matrix", "feature"]:
                            try:
                                st.write(f"Plot: {plot_type}")
                                plot_model(best_model, plot=plot_type, display_format="streamlit")
                            except Exception:
                                continue
                    else:
                        try:
                            st.write("Cluster Plot:")
                            plot_model(best_model, display_format="streamlit")
                        except Exception:
                            pass
                    st.markdown("---")
                    st.download_button("Download Leaderboard as CSV", leaderboard.to_csv(index=False), "leaderboard.csv")
                except Exception as e:
                    st.error(f"AutoML failed: {e}")
    
    # What-If Scenario Simulator Tab
    with tabs[2]:
        st.subheader("🤔 What-If Scenario Simulator")
        st.markdown("Simulate changes in feature values and see how your trained model would respond.")
        st.info("First, complete Data Cleaning and AutoML to use this feature.")
        import os
        model_path = "best_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                best_model = pickle.load(f)
            st.success("Loaded trained model! Now select feature values to simulate predictions.")
            sim_features = [col for col in cleaned_df.columns if cleaned_df[col].dtype in [np.float64, np.int64] or cleaned_df[col].dtype == object]
            input_data = {}
            for feat in sim_features:
                if pd.api.types.is_numeric_dtype(cleaned_df[feat]):
                    min_val = float(cleaned_df[feat].min())
                    max_val = float(cleaned_df[feat].max())
                    mean_val = float(cleaned_df[feat].mean())
                    input_data[feat] = st.slider(f"{feat}", min_value=min_val, max_value=max_val, value=mean_val)
                else:
                    options = cleaned_df[feat].dropna().unique().tolist()
                    if options:
                        input_data[feat] = st.selectbox(f"{feat}", options)
            if st.button("Simulate Prediction"):
                input_df = pd.DataFrame([input_data])
                result = None
                try:
                    if predict_model_cls is not None:
                        result = predict_model_cls(best_model, data=input_df)
                    elif predict_model_reg is not None:
                        result = predict_model_reg(best_model, data=input_df)
                except Exception:
                    try:
                        if predict_model_reg is not None:
                            result = predict_model_reg(best_model, data=input_df)
                    except Exception:
                        st.error("Prediction failed. The model may not support this input.")
                if result is not None:
                    st.write("Prediction result:")
                    st.dataframe(result)
        else:
            st.warning("Train and save a model in the AutoML tab first.")

    # Forecast Playground Tab
    with tabs[3]:
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
