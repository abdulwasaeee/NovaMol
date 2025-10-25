# pipeline.py (Upgraded with Advanced Statistics

import json
import time
import requests # New import
# ... (all other imports remain the same)
import subprocess
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
# from chembl_webresource_client.new_client import new_client
import selfies as sf
# At the top of pipeline.py, with other rdkit imports
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
# Note: RDKit must be installed with contribs for SA_Score
import SA_Score
import streamlit as st

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
</style>
""", unsafe_allow_html=True)


# --- Helper Functions (Updated) ---
def update_progress(progress_file, step_name, current, total):
    """Writes progress to a shared log file as JSON."""
    if progress_file:
        progress = {"step": step_name, "current": current, "total": total}
        with open(progress_file, 'w') as f:
            json.dump(progress, f)

# --- NEW: Function to check for novelty against PubChem ---
def check_pubchem_novelty(smiles_list):
    """
    Checks a list of SMILES against the PubChem database.
    Returns the count of molecules that ARE found in PubChem.
    """
    print("\n--- Checking for true novelty against PubChem database ---")
    found_count = 0
    # Use tqdm for a progress bar in the terminal
    for smiles in tqdm(smiles_list, desc="Querying PubChem"):
        try:
            # PUG REST API endpoint for getting CIDs from SMILES
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/txt"
            res = requests.get(url, timeout=5) # 5 second timeout
            if res.status_code == 200 and res.text.strip():
                found_count += 1
        except requests.exceptions.RequestException:
            # Ignore network errors, etc.
            continue
    return found_count


# ... (get_chembl_data, smiles_to_graph, etc. are unchanged) ...
def get_chembl_data(target_id='CHEMBL203', min_pchembl=5.0):
    # This function is unchanged
    print(f"--- Downloading data for target {target_id} from ChEMBL ---")
    # activity = new_client.activity
    # res = activity.filter(target_chembl_id=target_id, standard_type="IC50", pchembl_value__gte=min_pchembl)
    # df = pd.DataFrame(res)
    df = pd.read_csv(f'data/{target_id}_raw_chembl.csv')

    if df.empty:
        return df

    df = df[['canonical_smiles', 'pchembl_value']].dropna().drop_duplicates(subset=['canonical_smiles'])
    df['pchembl_value'] = pd.to_numeric(df['pchembl_value'])
    print(f"Downloaded and cleaned {len(df)} unique, active compounds.")

    molecules = [Chem.MolFromSmiles(smi) for smi in df['canonical_smiles']]
    df['logp'] = [Descriptors.MolLogP(m) if m else None for m in molecules]
    df['molecular_weight'] = [Descriptors.MolWt(m) if m else None for m in molecules]

    df.dropna(inplace=True)
    df = df.rename(columns={'canonical_smiles': 'smiles'})
    print(f"Processed down to {len(df)} compounds with all required properties.")
    return df

def smiles_to_graph(smiles: str):
    # This function is unchanged
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atom_features = [[
        atom.GetAtomicNum(), atom.GetFormalCharge(), float(atom.GetHybridization()),
        float(atom.GetIsAromatic()), atom.GetTotalNumHs(), atom.GetTotalValence()
    ] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    if mol.GetNumBonds() > 0:
        row, col = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row.extend([start, end]); col.extend([end, start])
        edge_index = torch.tensor([row, col], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def evaluate_multitask_gnn(loader, model, scalers, device, properties_to_predict):
    # This function is unchanged from the last version
    model.eval()
    all_predictions, all_targets, all_smiles = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            all_predictions.append(out.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            all_smiles.extend(batch.smiles)
            
    predictions_scaled = np.vstack(all_predictions)
    targets_scaled = np.vstack(all_targets)
    
    predictions_real = np.zeros_like(predictions_scaled)
    targets_real = np.zeros_like(targets_scaled)

    maes = {}
    for i, prop in enumerate(properties_to_predict):
        predictions_real[:, i] = scalers[prop].inverse_transform(predictions_scaled[:, i].reshape(-1, 1)).flatten()
        targets_real[:, i] = scalers[prop].inverse_transform(targets_scaled[:, i].reshape(-1, 1)).flatten()
        maes[prop] = np.mean(np.abs(predictions_real[:, i] - targets_real[:, i]))

    proof_data = {'SMILES': all_smiles[:5]}
    for i, prop in enumerate(properties_to_predict):
        proof_data[f'Actual {prop}'] = [f"{x:.2f}" for x in targets_real[:5, i]]
        proof_data[f'Predicted {prop}'] = [f"{x:.2f}" for x in predictions_real[:5, i]]
    
    proof_df = pd.DataFrame(proof_data)
    
    return maes, proof_df

def suggest_drug_candidate_potential(predicted_properties: dict):
    # This function is unchanged
    insights = []
    flags = []
    pchembl = predicted_properties.get('pchembl_value', 0)
    if pchembl >= 8.0: insights.append("Exceptional Potency")
    elif pchembl >= 7.0: insights.append("High Potency")
    elif pchembl >= 6.0: insights.append("Good Potency")
    elif pchembl >= 5.0: insights.append("Active")
    else: flags.append("Low Potency")

    logp = predicted_properties.get('logp', 99)
    mw = predicted_properties.get('molecular_weight', 999)
    if mw > 500: flags.append("High MW (>500)")
    elif mw < 300: insights.append("Fragment-like Size")
    else: insights.append("Ideal Drug-like Size")

    if logp > 5.0: flags.append("Poor Solubility (logP>5)")
    elif logp < 0: flags.append("Too Polar (logP<0)")
    elif 1.0 <= logp <= 3.0: insights.append("Optimal Solubility")
    else: insights.append("Acceptable Solubility")

    final_assessment = ", ".join(insights)
    if flags: final_assessment += " | FLAGS: " + ", ".join(flags)
    return final_assessment if final_assessment else "General Bioactive Compound"


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class MultiTaskGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=4):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, out_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return self.out(x)

class SELFIES_RNN(nn.Module):
    def __init__(self, vocab_size, emb_size=128, hidden_size=512, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        emb = self.embedding(x); out, hidden = self.rnn(emb, hidden)
        return self.fc(out), hidden

def sample_selfies(model, token2idx, idx2token, device, max_len=50, temperature=1.0):
    model.eval()
    start_token = '[C]'
    x = torch.tensor([[token2idx[start_token]]], device=device)
    hidden = None
    tokens = [start_token]
    for _ in range(max_len):
        out, hidden = model(x, hidden)
        probs = F.softmax(out.squeeze() / temperature, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        if idx == 0: break
        tokens.append(idx2token[idx])
        x = torch.tensor([[idx]], device=device)
    try: return sf.decoder(''.join(tokens))
    except: return None
    

# --- Main Pipeline Orchestrator (Updated) ---
def run_complete_pipeline(target_id: str, output_dir: str = 'results', progress_file: str = None):
    # ... (function signature and initial setup is the same)
    update_progress(progress_file, "Starting Pipeline...", 0, 100)
    print(f"--- Starting NovaMol Pipeline for Target: {target_id} ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- Configuration ---
    PROPERTIES_TO_PREDICT = ['pchembl_value', 'logp', 'molecular_weight']
    N_PROPERTIES = len(PROPERTIES_TO_PREDICT)
    N_MOLECULES_GNN = 10000
    N_MOLECULES_RNN = 2000
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    N_EPOCHS_GNN = 500
    N_EPOCHS_RNN = 5
    GENERATION_ATTEMPTS = 200
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # --- NEW: Dictionary to hold run statistics ---
    run_stats = {
        'gnn_epochs': N_EPOCHS_GNN,
        'rnn_epochs': N_EPOCHS_RNN,
        'generation_attempts': GENERATION_ATTEMPTS
    }

    # --- 1. Data Acquisition and Preparation (Unchanged) ---
    update_progress(progress_file, "Downloading Data from ChEMBL...", 5, 100)
    df_main = get_chembl_data(target_id=target_id)
    if len(df_main) < 100:
        raise ValueError(f"Insufficient data for target {target_id}. Found only {len(df_main)} compounds.")

    update_progress(progress_file, "Preparing Data...", 10, 100)
    print("\n--- 1. Preparing Data for Multi-Task GNN ---")
    gnn_data_list = []
    subset_df_gnn = df_main.head(N_MOLECULES_GNN)
    for _, row in tqdm(subset_df_gnn.iterrows(), total=subset_df_gnn.shape[0], desc="Creating GNN graphs"):
        graph = smiles_to_graph(row['smiles'])
        if graph:
            graph.y = torch.tensor([[row[p] for p in PROPERTIES_TO_PREDICT]], dtype=torch.float)
            graph.smiles = row['smiles']
            gnn_data_list.append(graph)

    train_val_data, test_data = train_test_split(gnn_data_list, test_size=0.15, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.17, random_state=42)

    scalers = {}
    for i, prop in enumerate(PROPERTIES_TO_PREDICT):
        targets = np.array([d.y[0, i].item() for d in train_data]).reshape(-1, 1)
        scalers[prop] = StandardScaler().fit(targets)
        for d in train_val_data + test_data:
            d.y[0, i] = torch.tensor(scalers[prop].transform([[d.y[0, i].item()]])[0,0], dtype=torch.float32)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    print(f"Data split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test.")

    # --- 2. GNN Training (Unchanged) ---
    print("\n--- 2. Training Multi-Task Predictive GNN ---")
    gnn_model = MultiTaskGNN(in_dim=train_data[0].num_node_features, out_dim=N_PROPERTIES).to(DEVICE)
    optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=LEARNING_RATE)
    loss_fn_gnn = nn.MSELoss()
    best_val_mae_sum = float('inf')
    best_model_path = os.path.join(output_dir, f'best_gnn_model_{target_id}.pth')

    for epoch in range(1, N_EPOCHS_GNN + 1):
        update_progress(progress_file, "Training GNN...", epoch, N_EPOCHS_GNN)
        gnn_model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE); optimizer_gnn.zero_grad()
            out = gnn_model(batch)
            loss = loss_fn_gnn(out, batch.y.to(DEVICE))
            loss.backward(); optimizer_gnn.step()
            
        val_maes, _ = evaluate_multitask_gnn(val_loader, gnn_model, scalers, DEVICE, PROPERTIES_TO_PREDICT)
        if sum(val_maes.values()) < best_val_mae_sum:
            best_val_mae_sum = sum(val_maes.values())
            torch.save(gnn_model.state_dict(), best_model_path)
        print(f"GNN Epoch {epoch:02d} | Val MAEs: " + ", ".join([f"{k[:10]}={v:.3f}" for k,v in val_maes.items()]))
    
    gnn_model.load_state_dict(torch.load(best_model_path))

    # --- 3. RNN Training (Unchanged) ---
    print("\n--- 3. Training Generative RNN ---")
    selfies_list = [sf.encoder(smi) for smi in tqdm(df_main['smiles'].head(N_MOLECULES_RNN), desc="Encoding to SELFIES") if smi and sf.encoder(smi)]
    all_tokens = set(t for s in selfies_list for t in sf.split_selfies(s))
    token2idx = {t: i + 1 for i, t in enumerate(sorted(all_tokens))}; token2idx['<PAD>'] = 0
    idx2token = {i: t for t, i in token2idx.items()}; vocab_size = len(token2idx)
    max_len = max(len(list(sf.split_selfies(s))) for s in selfies_list)
    selfies_tensor = torch.stack([torch.tensor([token2idx.get(t, 0) for t in list(sf.split_selfies(s))] + [0] * (max_len - len(list(sf.split_selfies(s)))), dtype=torch.long) for s in selfies_list])
    
    rnn_dataset = torch.utils.data.TensorDataset(selfies_tensor[:, :-1], selfies_tensor[:, 1:])
    rnn_loader = torch.utils.data.DataLoader(rnn_dataset, batch_size=128, shuffle=True)
    rnn_model = SELFIES_RNN(vocab_size).to(DEVICE)
    opt_rnn = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)
    
    for epoch in range(1, N_EPOCHS_RNN + 1):
        update_progress(progress_file, "Training RNN...", epoch, N_EPOCHS_RNN)
        for x, y in rnn_loader:
            x, y = x.to(DEVICE), y.to(DEVICE); opt_rnn.zero_grad()
            out, _ = rnn_model(x); loss = F.cross_entropy(out.reshape(-1, vocab_size), y.reshape(-1), ignore_index=0)
            loss.backward(); opt_rnn.step()
        print(f"RNN Epoch {epoch:02d}, Loss: {loss.item():.4f}")

    # --- 4. Final Evaluation & Report Generation ---
    update_progress(progress_file, "Final Evaluation...", 80, 100)
    print("\n--- A. GNN PERFORMANCE ON UNSEEN TEST DATA ---")
    test_maes_report, proof_df = evaluate_multitask_gnn(test_loader, gnn_model, scalers, DEVICE, PROPERTIES_TO_PREDICT)
    
    # In pipeline.py, replace the final analysis block inside run_complete_pipeline

    update_progress(progress_file, "Generating & Analyzing Molecules...", 90, 100)
    print("\n--- B. GENERATING AND ANALYZING NOVEL MOLECULES ---")
    analysis_results = []
    try:
        generated_smiles = [sample_selfies(rnn_model, token2idx, idx2token, DEVICE, temperature=0.95) for _ in range(GENERATION_ATTEMPTS)]
        valid_smiles = [s for s in generated_smiles if s and Chem.MolFromSmiles(s)]
        novel_smiles = [s for s in valid_smiles if s not in df_main['smiles'].values]
        
        run_stats['molecules_valid_syntax'] = len(valid_smiles)
        run_stats['molecules_novel_vs_training_set'] = len(novel_smiles)
        print(f"Generated {len(novel_smiles)} valid and novel molecules.")

        if novel_smiles:
            update_progress(progress_file, "Checking PubChem Novelty...", 95, 100)
            count_in_pubchem = check_pubchem_novelty(novel_smiles)
            run_stats['molecules_found_in_pubchem'] = count_in_pubchem

            novel_graphs = [smiles_to_graph(s) for s in novel_smiles]
            valid_novel_data = [(smi, g) for smi, g in zip(novel_smiles, novel_graphs) if g is not None]

            if valid_novel_data:
                smiles_for_analysis, graphs_for_analysis = zip(*valid_novel_data)
                
                predict_loader = DataLoader(list(graphs_for_analysis), batch_size=len(graphs_for_analysis))
                batch = next(iter(predict_loader)).to(DEVICE)
                preds_scaled = gnn_model(batch).cpu().detach().numpy()

                for i, smiles in enumerate(smiles_for_analysis):
                    mol = Chem.MolFromSmiles(smiles)
                    
                    # Existing calculated features
                    tpsa = Descriptors.TPSA(mol)
                    qed_score = QED.qed(mol)
                    sa_score = SA_Score.calculateScore(mol)
                    
                    # --- NEW: Calculate Aromatic Ring Count and Fsp3 ---
                    aromatic_rings = Descriptors.NumAromaticRings(mol)
                    fsp3 = Descriptors.FractionCSP3(mol)

                    # Get predicted properties from the GNN
                    predicted_props = {prop: scalers[prop].inverse_transform(preds_scaled[i, j].reshape(1, -1))[0,0] for j, prop in enumerate(PROPERTIES_TO_PREDICT)}
                    assessment = suggest_drug_candidate_potential(predicted_props)
                    
                    # --- NEW: Append all features to the results list ---
                    analysis_results.append([
                        smiles,
                        f"{predicted_props['pchembl_value']:.2f}",
                        f"{predicted_props['logp']:.2f}",
                        f"{predicted_props['molecular_weight']:.2f}",
                        f"{tpsa:.2f}",
                        f"{qed_score:.2f}",
                        f"{sa_score:.2f}",
                        aromatic_rings, # New
                        f"{fsp3:.2f}",   # New
                        assessment
                    ])
    except (ValueError, IndexError) as e:
        print(f"\nWARNING: Could not analyze generated molecules. Details: {e}")

    # --- 6. Save results to CSV (with new headers) ---
    update_progress(progress_file, "Saving Results...", 99, 100)
    # --- NEW: Add the new column names to the headers list ---
    headers = [
        "Novel SMILES", "Pred. pChEMBL", "Pred. logP", "Pred. MW",
        "TPSA", "QED", "SA_Score", "Aromatic Rings", "Fsp3", 
        "Drug Candidate Assessment"
    ]
    df_analysis = pd.DataFrame(analysis_results, columns=headers)
    csv_output_path = os.path.join(output_dir, f"{target_id}_novel_molecules.csv")
    df_analysis.to_csv(csv_output_path, index=False)
    
    update_progress(progress_file, "Complete", 100, 100)
    print(f"\n--- Pipeline for Target {target_id} Complete ---")
    
    return run_stats, test_maes_report, proof_df, csv_output_path
