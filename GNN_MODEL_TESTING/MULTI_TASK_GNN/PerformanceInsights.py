import os
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
# from chembl_webresource_client.new_client import new_client
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit.Chem import Descriptors

PROPERTIES_TO_PREDICT = ['pchembl_value', 'logp', 'molecular_weight']
N_MOLECULES_GNN = 10000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
N_EPOCHS_GNN = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 1. Deeper GIN with Dropout
class MultiTaskGNN_v1(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=3):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # mean instead of sum
        x = F.relu(self.lin1(x))
        return self.lin2(x)

#2. Add a Third GIN Layer
class MultiTaskGNN_v2(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=3):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in [self.conv1, self.conv2, self.conv3]:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x)

# 3. Residual Connections
class MultiTaskGNN_v3(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=3):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        x = h1 + h2  # residual
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x)

# 4. GCN Layer Instead of One GIN
class MultiTaskGNN_v4(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=3):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # hybrid GIN + GCN
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x)

# 5. Two-Layer MLP Head
class MultiTaskGNN_v5(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=3):
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

class MultiTaskGNN_v6(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 3, num_layers: int = 3, hidden_dim: int = 128, dropout_rate: float = 0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.initial_embedding = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.initial_embedding(x)
        for conv, bn in zip(self.convs, self.batch_norms):
            x_residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = x + x_residual
        x_graph = global_add_pool(x, batch)
        output = self.mlp(x_graph)

        return output

def get_chembl_data(target_id='CHEMBL203', min_pchembl=5.0):

    data_dir: str = 'data'
    os.makedirs(data_dir, exist_ok=True)

    print(f"--- Downloading data for target {target_id} from ChEMBL ---")
    # activity = new_client.activity
    # res = activity.filter(target_chembl_id=target_id, standard_type="IC50", pchembl_value__gte=min_pchembl)
    # df = pd.DataFrame(res)
    df = pd.read_csv(f'data/{target_id}_raw_chembl.csv')

    # raw_csv_path = os.path.join(data_dir, f"{target_id}_raw_chembl.csv")
    # df.to_csv(raw_csv_path, index=False)
    # print(f"Raw ChEMBL data saved to {raw_csv_path}")

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

    processed_csv_path = os.path.join(data_dir, f"{target_id}_processed_analysis.csv")
    df.to_csv(processed_csv_path, index=False)
    print(f"Processed analysis results saved to {processed_csv_path}")

    return df

def smiles_to_graph(smiles: str):
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

    metrics = {}
    for i, prop in enumerate(properties_to_predict):
        predictions_real[:, i] = scalers[prop].inverse_transform(predictions_scaled[:, i].reshape(-1, 1)).flatten()
        targets_real[:, i] = scalers[prop].inverse_transform(targets_scaled[:, i].reshape(-1, 1)).flatten()
        mae = mean_absolute_error(targets_real[:, i], predictions_real[:, i])
        mse = mean_squared_error(targets_real[:, i], predictions_real[:, i])
        rmse = mse ** 0.5
        r2 = r2_score(targets_real[:, i], predictions_real[:, i])
        metrics[prop] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

    proof_data = {'SMILES': all_smiles[:5]}
    for i, prop in enumerate(properties_to_predict):
        proof_data[f'Actual {prop}'] = [f"{x:.2f}" for x in targets_real[:5, i]]
        proof_data[f'Predicted {prop}'] = [f"{x:.2f}" for x in predictions_real[:5, i]]

    proof_df = pd.DataFrame(proof_data)

    return metrics, proof_df

def flatten_metrics(val_metrics: dict, flat=None) -> dict:
    if flat is None:
        flat = {}
    for prop, metrics in val_metrics.items():
        for metric_name, value in metrics.items():
            flat[f"{metric_name}_{prop}"] = value
    return flat


def print_validation_metrics(val_metrics):
    """
    Prints validation metrics for each property in a formatted way.
    """
    for prop, metrics in val_metrics.items():
        metrics_str = " | ".join([f"{metric_name}: {value:.4f}" for metric_name, value in metrics.items()])
        print(f"{prop}: {metrics_str}")
    print()


def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN settings: deterministic True, benchmark False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_data_for_gnn(target_id: str, properties_to_predict: list | None = None):
    print(f"--- Starting NovaMol Pipeline for Target: {target_id} ---")
    # --- Configuration ---
    if properties_to_predict is None:
        properties_to_predict = PROPERTIES_TO_PREDICT

    df_main = get_chembl_data(target_id=target_id)
    if len(df_main) < 100:
        raise ValueError(f"Insufficient data for target {target_id}. Found only {len(df_main)} compounds.")

    print("\n--- 1. Preparing Data for Multi-Task GNN ---")
    gnn_data_list = []
    subset_df_gnn = df_main.head(N_MOLECULES_GNN)
    for _, row in tqdm(subset_df_gnn.iterrows(), total=subset_df_gnn.shape[0], desc="Creating GNN graphs"):
        graph = smiles_to_graph(row['smiles'])
        if graph:
            graph.y = torch.tensor([[row[p] for p in properties_to_predict]], dtype=torch.float)
            graph.smiles = row['smiles']
            gnn_data_list.append(graph)

    train_val_data, test_data = train_test_split(gnn_data_list, test_size=0.15, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.17, random_state=42)

    scalers = {}
    for i, prop in enumerate(properties_to_predict):
        targets = np.array([d.y[0, i].item() for d in train_data]).reshape(-1, 1)
        scalers[prop] = StandardScaler().fit(targets)
        for d in train_val_data + test_data:
            d.y[0, i] = torch.tensor(scalers[prop].transform([[d.y[0, i].item()]])[0, 0], dtype=torch.float32)

    gen = torch.Generator()
    gen.manual_seed(42)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, generator=gen,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, generator=torch.Generator().manual_seed(42),
                            worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                             generator=torch.Generator().manual_seed(42), worker_init_fn=worker_init_fn)

    print(f"Data split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test.")
    return train_loader, val_loader, test_loader, train_data, scalers


def run_gnn(target_id: str, gnn_class = MultiTaskGNN_v6, properties_to_predict: list | None = None, train_loader=None, val_loader=None, test_loader=None, train_data=None, scalers=None):

    if properties_to_predict is None:
        properties_to_predict = PROPERTIES_TO_PREDICT

    # --- 2. GNN Training ---
    print("\n--- 2. Training Multi-Task Predictive GNN ---")
    gnn_model = gnn_class(in_dim=train_data[0].num_node_features, out_dim=len(properties_to_predict)).to(DEVICE)
    optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=LEARNING_RATE)
    loss_fn_gnn = nn.MSELoss()
    best_val_mae_sum = float('inf')

    model_results_dir = 'model_results'
    os.makedirs(model_results_dir, exist_ok=True)
    best_model_path = os.path.join(model_results_dir, f'best_gnn_model_{target_id}.pth')

    training_results_dir = 'training_results'
    os.makedirs(training_results_dir, exist_ok=True)

    test_results_dir = 'test_results'
    os.makedirs(test_results_dir, exist_ok=True)

    metrics_history = []
    for epoch in range(1, N_EPOCHS_GNN + 1):
        gnn_model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE); optimizer_gnn.zero_grad()
            out = gnn_model(batch)
            loss = loss_fn_gnn(out, batch.y.to(DEVICE))
            loss.backward(); optimizer_gnn.step()

        val_metrics, _ = evaluate_multitask_gnn(val_loader, gnn_model, scalers, DEVICE, properties_to_predict)

        # Flatten metrics for CSV
        epoch_metrics = {'epoch': epoch}
        metrics_history.append(flatten_metrics(val_metrics, epoch_metrics))

        mae_sum = sum([metrics["MAE"] for metrics in val_metrics.values()])
        if mae_sum < best_val_mae_sum:
            best_val_mae_sum = mae_sum
            torch.save(gnn_model.state_dict(), best_model_path)
        print(f"\nGNN Epoch {epoch:02d} | Validation Metrics:")
        print_validation_metrics(val_metrics)

    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(os.path.join(training_results_dir, f"{target_id}_{gnn_model.__class__.__name__}_val_metrics_history.csv"), index=False)
    gnn_model.load_state_dict(torch.load(best_model_path))

    print("\n--- A. GNN PERFORMANCE ON UNSEEN TEST DATA ---")
    test_maes_report, proof_df = evaluate_multitask_gnn(test_loader, gnn_model, scalers, DEVICE, properties_to_predict)
    print_validation_metrics(test_maes_report)
    data_for_df = {k: [v] for k, v in flatten_metrics(test_maes_report).items()}
    test_metrics_df = pd.DataFrame(data_for_df)
    test_metrics_df.to_csv(os.path.join(test_results_dir, f"{target_id}_{gnn_model.__class__.__name__}_test_metrics.csv"), index=False)

    return test_maes_report, proof_df


def run_all_gnn(dataset = 'CHEMBL203', properties_to_predict: list | None = None):

    training_results_dir = 'training_results'
    test_results_dir = 'test_results'

    gnn_classes = [
        MultiTaskGNN_v1,
        MultiTaskGNN_v2,
        MultiTaskGNN_v3,
        MultiTaskGNN_v4,
        MultiTaskGNN_v5,
        MultiTaskGNN_v6
    ]
    if properties_to_predict is None:
        properties_to_predict = PROPERTIES_TO_PREDICT

    train_loader, val_loader, test_loader, train_data, scalers = prepare_data_for_gnn(target_id=dataset, properties_to_predict=properties_to_predict)

    for gnn_cls in gnn_classes:
        print(f"\n--- Running pipeline for {gnn_cls.__name__} ---")
        run_gnn(dataset, gnn_class=gnn_cls, properties_to_predict=properties_to_predict, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, train_data=train_data, scalers=scalers)


# To read and visualize the results:
    df_gnn_v1 = pd.read_csv(f'{training_results_dir}/{dataset}_MultiTaskGNN_v1_val_metrics_history.csv')
    df_gnn_v2 = pd.read_csv(f'{training_results_dir}/{dataset}_MultiTaskGNN_v2_val_metrics_history.csv')
    df_gnn_v3 = pd.read_csv(f'{training_results_dir}/{dataset}_MultiTaskGNN_v3_val_metrics_history.csv')
    df_gnn_v4 = pd.read_csv(f'{training_results_dir}/{dataset}_MultiTaskGNN_v4_val_metrics_history.csv')
    df_gnn_v5 = pd.read_csv(f'{training_results_dir}/{dataset}_MultiTaskGNN_v5_val_metrics_history.csv')
    df_gnn_v6 = pd.read_csv(f'{training_results_dir}/{dataset}_MultiTaskGNN_v6_val_metrics_history.csv')

    # Define properties and measurements
    properties = properties_to_predict if properties_to_predict else PROPERTIES_TO_PREDICT
    # properties = ['pchembl_value', 'logp', 'molecular_weight']
    measurements = ['MAE', 'MSE', 'RMSE', 'R2']
    gnn_labels = ['GNN v1', 'GNN v2', 'GNN v3', 'GNN v4', 'GNN v5', 'GNN v6']
    dfs = [df_gnn_v1, df_gnn_v2, df_gnn_v3, df_gnn_v4, df_gnn_v5, df_gnn_v6]

    fig, axes = plt.subplots(len(properties), len(measurements), figsize=(20, 12), sharex=True)
    axes = np.atleast_2d(axes)  # ensure 2-D indexing works even when a dimension == 1
    for i, prop in enumerate(properties):
        for j, metric in enumerate(measurements):
            ax = axes[i, j]
            for df, label in zip(dfs, gnn_labels):
                col = f'{metric}_{prop}'
                if col in df.columns:
                    ax.plot(df['epoch'], df[col], label=label)
            if i == 0:
                ax.set_title(metric)
            if j == 0:
                ax.set_ylabel(prop)
            if i == 0 and j == 0:
                ax.legend()
            ax.grid()

    plt.tight_layout()
    plt.show()

    files = [
        f"{test_results_dir}/{dataset}_MultiTaskGNN_v1_test_metrics.csv",
        f"{test_results_dir}/{dataset}_MultiTaskGNN_v2_test_metrics.csv",
        f"{test_results_dir}/{dataset}_MultiTaskGNN_v3_test_metrics.csv",
        f"{test_results_dir}/{dataset}_MultiTaskGNN_v4_test_metrics.csv",
        f"{test_results_dir}/{dataset}_MultiTaskGNN_v5_test_metrics.csv",
        f"{test_results_dir}/{dataset}_MultiTaskGNN_v6_test_metrics.csv"
    ]
    models = [f.split('_v')[1].split('_test')[0] for f in files]

    all_dfs = []
    for file, model in zip(files, models):
        try:
            df = pd.read_csv(file)
            df['Model'] = f'v{model}'
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File not found: {file}. Skipping.")
            continue

    if not all_dfs:
        raise Exception("Error: No dataframes were loaded. Check file paths/names.")

    df_combined = pd.concat(all_dfs, ignore_index=True)

    # 2. Reshape to Long Format (Melt)
    id_vars = ['Model']
    value_vars = [col for col in df_combined.columns if col != 'Model']

    df_long = pd.melt(
        df_combined,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='Metric',
        value_name='Value'
    )

    # 3. Split the 'Metric' column into Metric_Type and Property
    df_long[['Metric_Type', 'Property']] = df_long['Metric'].str.split('_', n=1, expand=True)

    # Clean up Property names for better labels
    property_map = {
        'pchembl_value': 'pChEMBL Value',
        'logp': 'LogP',
        'molecular_weight': 'Molecular Weight'
    }
    df_long['Property_Label'] = df_long['Property'].map(property_map)


    print("Data transformation complete. Ready for plotting.")
    print("-" * 50)


    # 4. Visualization: Grouped Bar Charts by Property

    # Define properties and metrics for iteration
    properties = df_long['Property'].unique()
    metrics_to_plot = ['MAE', 'MSE', 'RMSE', 'R2']

    # Create a figure with subplots for each property (3 rows, 1 column)
    fig, axes = plt.subplots(len(properties), 1, figsize=(12, 18), sharex=False)
    axes = np.atleast_1d(axes)  # ensure axes[i] works when len(properties) == 1
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.subplots_adjust(hspace=0.45)

    model_order = [f'v{j}' for j in range(1, 7)]

    for i, prop in enumerate(properties):
        df_prop = df_long[df_long['Property'] == prop]
        df_prop_filtered = df_prop[df_prop['Metric_Type'].isin(metrics_to_plot)].copy()
        ax = axes[i]
        sns.barplot(
            x='Metric_Type',
            y='Value',
            hue='Model',
            data=df_prop_filtered,
            palette='magma',
            ax=ax,
            order=['MAE', 'MSE', 'RMSE', 'R2'],
            hue_order=model_order,
            errorbar=None
        )
        property_label = df_prop_filtered['Property_Label'].iloc[0]
        ax.set_title(f'Model Performance Comparison for: {property_label}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Metric Type', fontsize=14)
        ax.set_ylabel(f'Metric Value for {property_label}', fontsize=14)
        ax.legend(title='Model Version', loc='upper right')
        ax.text(
            0.98,
            0.98,
            r'Goal: $\mathbf{\downarrow}$ for MAE/MSE/RMSE, $\mathbf{\uparrow}$ for $\mathbf{R}^2$',
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7, ec="gray")
        )

    plt.show()

    # --- OPTIONAL: HEATMAP CODE ---
    print("-" * 50)
    print("Generating Optional Heatmap...")

    # Pivot the data back to wide format, but keep only Model and Metric/Value
    df_heatmap = df_long.pivot(index='Model', columns='Metric', values='Value')

    # Order the models for the heatmap rows
    df_heatmap = df_heatmap.reindex(model_order)

    # Order the columns (metrics) logically: pchembl, then logp, then mol_weight
    metric_columns_order = [
        'MAE_pchembl_value', 'MSE_pchembl_value', 'RMSE_pchembl_value', 'R2_pchembl_value',
        'MAE_logp', 'MSE_logp', 'RMSE_logp', 'R2_logp',
        'MAE_molecular_weight', 'MSE_molecular_weight', 'RMSE_molecular_weight', 'R2_molecular_weight'
    ]

    cols_present = [c for c in metric_columns_order if c in df_heatmap.columns]
    missing_cols = [c for c in metric_columns_order if c not in df_heatmap.columns]

    if missing_cols:
        print(f"Warning: missing heatmap columns skipped: {missing_cols}")

    if not cols_present:
        raise Exception("No metric columns available for heatmap. Skipping heatmap generation.")

    df_heatmap = df_heatmap[cols_present]

    plt.figure(figsize=(15, 6))

    # IMPORTANT: R2 requires a reversed color scale (higher is better) compared to errors (lower is better).
    # We'll use a single color map and annotate the plot heavily.

    sns.heatmap(
        df_heatmap,
        annot=True,          # Display the metric values on the cells
        fmt=".3f",           # Format to 3 decimal places
        cmap='RdYlBu_r',     # Use a diverging color map, reversed so red is lower (worse error, better R2)
        linewidths=.5,
        linecolor='black',
        cbar_kws={'label': 'Metric Value'}
    )

    plt.title('Comprehensive Model Performance Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Metric (Columns are ordered by Property)', fontsize=14)
    plt.ylabel('Model Version', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


    # --- 5. NORMALIZATION AND AVERAGING ---

    # 5a. Normalization Function
    # Maps the best score to 1 and the worst score to 0.
    def normalize_performance(series, metric_type):
        """
        Normalizes a metric series across all models (0=Worst, 1=Best).

        Args:
            series (pd.Series): The column of values for a specific metric (e.g., MAE_logp).
            metric_type (str): The type of metric (e.g., 'MAE', 'R2').
        """
        min_val = series.min()
        max_val = series.max()

        if max_val == min_val:
            # If all models have the same score, they all get 1.0 (perfect relative score)
            return pd.Series([1.0] * len(series), index=series.index)

        normalized = (series - min_val) / (max_val - min_val)

        # Error metrics (MAE, MSE, RMSE): Lower is better, so invert the score (1 - normalized)
        if metric_type in ['MAE', 'MSE', 'RMSE']:
            return 1 - normalized
        else:
            # R2: Higher is better, so use the direct normalized score
            return normalized

    # 5b. Apply Normalization
    # Normalize the 'Value' column, grouping by the specific metric column ('Metric_Type', 'Property').
    # This ensures MAE_pchembl_value is normalized only against other MAE_pchembl_value scores.
    df_long['Normalized_Value'] = df_long.groupby(['Metric_Type', 'Property'])['Value'].transform(
        lambda x: normalize_performance(x, x.name[0])
    )

    # 5c. Calculate Average Normalized Score
    # Now, average the Normalized_Value across all three properties for each Model/Metric_Type combination.
    df_avg_performance = df_long.groupby(['Model', 'Metric_Type'])['Normalized_Value'].mean().reset_index()

    print("-" * 50)
    print("Average Normalized Performance Data:")
    print(df_avg_performance)
    print("-" * 50)

    # 6. VISUALIZATION: AVERAGE NORMALIZED PERFORMANCE

    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Order the models for plotting: best-performing model first (highest average normalized score)
    # This provides the overall ranking.
    model_order_sum = (
        df_avg_performance.groupby('Model')['Normalized_Value']
        .mean()
        .sort_values(ascending=False)
        .index
    )

    sns.barplot(
        x='Metric_Type',
        y='Normalized_Value',
        hue='Model',
        data=df_avg_performance,
        order=['MAE', 'MSE', 'RMSE', 'R2'], # Consistent metric order
        hue_order=model_order_sum,           # Use the ranking order
        palette='Spectral',                  # Distinct colors
        errorbar=None
    )

    # # Add a reference line at 0.5 (average performance)
    # plt.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Average Performance (0.5)')

    plt.title('Average Normalized Performance Across All Multi-Task Properties', fontsize=16, fontweight='bold')
    plt.xlabel('Metric Type', fontsize=14)
    plt.ylabel('Average Normalized Score (1.0 = Best, 0.0 = Worst)', fontsize=14)
    plt.legend(title='Model Version', loc='upper left')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------------
    # --- BONUS PLOT: OVERALL AVERAGE SCORE ---
    # To determine the SINGLE BEST MODEL with one bar per model.

    df_overall_avg = df_avg_performance.groupby('Model')['Normalized_Value'].mean().reset_index()
    df_overall_avg = df_overall_avg.sort_values('Normalized_Value', ascending=False)
    model_order_overall = df_overall_avg['Model']

    plt.figure(figsize=(9, 6))
    sns.barplot(
        x='Model',
        y='Normalized_Value',
        data=df_overall_avg,
        order=model_order_overall,
        hue='Model',
        palette='viridis'
    )

    # plt.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.title('Overall Multi-Task Performance Ranking', fontsize=16, fontweight='bold')
    plt.xlabel('Model Version', fontsize=14)
    plt.ylabel('Overall Average Normalized Score', fontsize=14)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()