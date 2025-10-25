# app.py (Upgraded with PubChem count in summary)

import streamlit as st
import pandas as pd
import pipeline
from concurrent.futures import ProcessPoolExecutor
import os
import time
import json
import uuid







# Purple Theme CSS - Add this to your existing pipeline UI
st.markdown("""
<style>
    /* Color Variables */
    :root {
        --primary-purple: #667eea;
        --secondary-purple: #764ba2;
        --dark-purple: #4a2c85;
        --light-purple: #9c88ff;
        --very-light-purple: #f8f6ff;
        --text-dark: #202124;
        --text-medium: #5f6368;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar arrow visibility fix */
    button[kind="header"] {
        background-color: var(--primary-purple) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    
    button[kind="header"]:hover {
        background-color: var(--secondary-purple) !important;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #f8f6ff 0%, #ffffff 50%, #f0efff 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, var(--primary-purple) 0%, var(--secondary-purple) 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white;
    }
    
    /* Headers - Purple */
    h1, h2, h3, h4, h5, h6 {
        color: var(--dark-purple) !important;
    }
    
    /* All text - Dark for visibility */
    p, div, span, li, label {
        color: var(--text-dark) !important;
    }
    
    /* Buttons - Purple gradient */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-purple), var(--secondary-purple)) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Input fields - Light purple background */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stMultiSelect > div > div {
        background: var(--very-light-purple) !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 8px !important;
        color: var(--text-dark) !important;
    }
    
    /* Progress bar - Purple gradient */
    .stProgress > div > div > div {
        background: var(--light-purple);
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%) !important;
        border-left: 4px solid #28a745 !important;
        color: var(--text-dark) !important;
    }
    
    /* Info messages */
    .stInfo {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%) !important;
        border-left: 4px solid var(--primary-purple) !important;
        color: var(--text-dark) !important;
    }
    
    /* Warning messages */
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%) !important;
        border-left: 4px solid #ffc107 !important;
        color: var(--text-dark) !important;
    }
    
    /* Error messages */
    .stError {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%) !important;
        border-left: 4px solid #dc3545 !important;
        color: var(--text-dark) !important;
    }
    
    /* Metrics - White cards with purple accent */
    [data-testid="metric-container"] {
        background: white !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.1) !important;
        border: 2px solid rgba(102, 126, 234, 0.1) !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2) !important;
        border-color: var(--light-purple) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--primary-purple) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-dark) !important;
        font-weight: 500 !important;
    }
    
    /* Dataframe - Purple border */
    .stDataFrame {
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
    }
    
    /* Tabs - Purple styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--very-light-purple);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--primary-purple) !important;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-purple), var(--secondary-purple));
        color: white !important;
        border-radius: 8px;
    }
    
    /* Expander - Purple header */
    .streamlit-expanderHeader {
        background: var(--very-light-purple) !important;
        color: var(--dark-purple) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    /* File uploader - Purple border */
    .stFileUploader {
        background: var(--very-light-purple) !important;
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Download button - Purple */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--primary-purple), var(--secondary-purple)) !important;
        color: white !important;
        border-radius: 10px !important;
    }
    
    /* Slider - Purple track */
    .stSlider > div > div > div > div {
        background: var(--primary-purple) !important;
    }
    
    /* Checkbox/Radio - Purple when selected */
    .stCheckbox > label > div[data-checked="true"],
    .stRadio > label > div[data-checked="true"] {
        background-color: var(--primary-purple) !important;
    }
    
            
    /* Sidebar purple background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar - make labels white */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Sidebar selectbox - WHITE BACKGROUND with BLACK TEXT */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        color: black !important;
        # background-color: var(--dark-purple) !important;
    }
    
    /* Dropdown menu options - WHITE BACKGROUND with BLACK TEXT */
    [data-testid="stSidebar"] [role="listbox"] {
        background-color: white !important;
    }
    
    [data-testid="stSidebar"] [role="option"] {
        background-color: white !important;
        color: black !important;
    }
    
    [data-testid="stSidebar"] [role="option"]:hover {
        background-color: #f0f0f0 !important;
        color: black !important;
    }
    
    /* Sidebar button - white with purple text */
    [data-testid="stSidebar"] .stButton button {
        background-color: white !important;
        color: var(--primary-purple) !important;
        border: none !important;
        font-weight: 600 !important;
    }
            
    .stSidebar p, div, span, li, label {
        color: white!important;
    }
    
    .stMainBlockContainer p, div, li, label {
        color: var(--text-dark)!important;
    }
            
    /* Progress bar - PURPLE */
    .stProgress > div > div > div > div {
        background-color: var(--primary-purple) !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Progress bar background - light purple */
    .stProgress > div > div {
        background-color: rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Main content - ensure text is visible */
    .main h1, .main h2, .main h3 {
        color: var(--dark-purple) !important;
    }
    
    .main p, .main div, .main span, .main li {
        color: #202124 !important;
    }
    
    /* Buttons purple gradient */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Download button purple */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    /* Metrics purple accent */
    [data-testid="stMetricValue"] {
        color: var(--primary-purple) !important;
    }
    
    /* Force select/dropdown background to light and text to black (global and sidebar) */
    .stSelectbox [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [role="listbox"],
    [role="option"],
    div[role="listbox"],
    div[role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Ensure dropdown toggle (selected value) is light */
    .stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] ,
    .stSelectbox [data-baseweb="select"] > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Hover and selected option styling for readability */
    [role="option"]:hover,
    [role="option"][aria-selected="true"],
    .stSelectbox [role="option"]:hover,
    .stSelectbox [role="option"][aria-selected="true"] {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }
    
    /* Cover BaseWeb/internal menu classes generically */
    [class*="listbox"], [class*="option"], [class*="menu"] [role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if 'job_future' not in st.session_state:
    st.session_state.job_future = None
if 'job_started' not in st.session_state:
    st.session_state.job_started = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'error' not in st.session_state:
    st.session_state.error = None
if 'progress_file' not in st.session_state:
    st.session_state.progress_file = ""

# --- Main App UI ---

st.title("💊 NovaMol: AI Drug Discovery Pipeline")
st.write("Select a biological target, and the pipeline will train predictive and generative models to discover novel, drug-like molecules.")

st.sidebar.header("Configuration")

TARGET_DATASETS = {
    "Lung Cancer (EGFR)": "CHEMBL203",
    "Colon Cancer (VEGFR2)": "CHEMBL240",
    "Breast Cancer (ERα)": "CHEMBL226",
    "Liver Cancer (c-Met)": "CHEMBL1978",
    "Antibacterial (DHFR)": "CHEMBL202",
}

selected_target_name = st.sidebar.selectbox(
    "Select a Target Dataset:",
    options=list(TARGET_DATASETS.keys())
)
target_id_input = TARGET_DATASETS[selected_target_name]

executor = ProcessPoolExecutor(max_workers=1)

if st.sidebar.button("🚀 Run Discovery Pipeline"):
    st.session_state.job_started = True
    st.session_state.results = None
    st.session_state.error = None
    
    st.session_state.progress_file = f"progress_{uuid.uuid4()}.log"
    
    output_dir = f'results_{target_id_input}'
    st.session_state.job_future = executor.submit(
        pipeline.run_complete_pipeline, 
        target_id=target_id_input,
        output_dir=output_dir,
        progress_file=st.session_state.progress_file
    )
    st.info(f"Pipeline started for {selected_target_name} ({target_id_input}). You can see detailed progress below.")

if st.session_state.job_started:
    future = st.session_state.job_future
    
    progress_bar = st.progress(0)
    progress_text = st.empty()

    while not future.done():
        if os.path.exists(st.session_state.progress_file):
            with open(st.session_state.progress_file, 'r') as f:
                try:
                    progress = json.load(f)
                    percent_complete = int((progress['current'] / progress['total']) * 100) if progress['total'] > 0 else 0
                    progress_bar.progress(percent_complete)
                    
                    if 'Epoch' in progress['step']:
                        progress_text.text(f"Status: {progress['step']} (Epoch {progress['current']}/{progress['total']})")
                    else:
                         progress_text.text(f"Status: {progress['step']}")
                
                except (json.JSONDecodeError, KeyError):
                    pass
        time.sleep(0.5)

    try:
        st.session_state.results = future.result()
        progress_bar.progress(100)
        progress_text.text("Pipeline Complete!")
    except Exception as e:
        st.session_state.error = e
        progress_text.error("An error occurred during the pipeline.")
    
    if os.path.exists(st.session_state.progress_file):
        os.remove(st.session_state.progress_file)

    if st.session_state.results:
        st.success(f"✅ Pipeline completed successfully for {selected_target_name}!")
        
        run_stats, performance_report, proof_df, csv_path = st.session_state.results

        # --- MODIFIED: Updated Run Summary Section ---
        st.header("1. Run Summary")
        
        truly_novel_count = run_stats.get('molecules_novel_vs_training_set', 0) - run_stats.get('molecules_found_in_pubchem', 0)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("GNN Epochs", run_stats.get('gnn_epochs', 'N/A'))
        col2.metric("RNN Epochs", run_stats.get('rnn_epochs', 'N/A'))
        col3.metric("Novel (vs. Training Set)", run_stats.get('molecules_novel_vs_training_set', 'N/A'))
        col4.metric("Found in PubChem", run_stats.get('molecules_found_in_pubchem', 'N/A'))
        col5.metric("Truly Novel", truly_novel_count)


        st.header("2. Model Performance Report")
        st.subheader("Aggregate Performance (Mean Absolute Error)")
        
        cols = st.columns(len(performance_report))
        for i, (metric, value) in enumerate(performance_report.items()):
            cols[i].metric(label=metric, value=f"{value:.4f}")

        st.subheader("Sample Predictions on Test Set (Proof)")
        st.dataframe(proof_df)

        st.header("3. Generated Novel Drug Candidates")
        try:
            df_results = pd.read_csv(csv_path)
            st.dataframe(df_results)
            
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=f,
                    file_name=os.path.basename(csv_path),
                    mime="text/csv",
                )
        except FileNotFoundError:
            st.error(f"Could not find the results file at {csv_path}.")
        except pd.errors.EmptyDataError:
             st.warning("The pipeline ran successfully, but no valid novel molecules were generated in this run.")
        
    elif st.session_state.error:
        st.error("An error occurred while running the pipeline:")
        st.exception(st.session_state.error)
