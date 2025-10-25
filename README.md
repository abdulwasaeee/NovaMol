
# 🧬 NovaMol: A Generative and Predictive AI Engine for Drug Discovery

**Contributors:** Ahmed Abdul Wasae, Raja Vijayakumar, John Joseph Peter, Hari Krishna

<img width="1782" height="735" alt="Screenshot 2025-10-10 142459" src="https://github.com/user-attachments/assets/319f0598-c5dd-44fa-9e11-a15109239f2c" />


NovaMol is an advanced machine learning pipeline that combines **Graph Neural Networks (GNNs)** for property prediction and **Recurrent Neural Networks (RNNs)** for molecule generation. It is designed to accelerate *de novo* drug discovery by exploring novel chemical space and providing instant analysis of generated molecules.

---

## 🚀 Core Features

* **Generative Chemistry:** Utilizes an LSTM-based RNN to learn the language of molecular structures and generate novel, chemically valid molecules in SMILES format.
* **Predictive Analytics:** Employs a powerful Graph Neural Network (GNN) to predict critical molecular properties directly from the graph structure of a molecule.
* **Targeted Discovery:** Can be trained on specific biological targets (e.g., EGFR for lung cancer) to generate and evaluate molecules with a higher probability of being effective.
* **Comprehensive Analysis:** Automatically calculates a suite of drug-likeness properties for each generated molecule, including TPSA, QED, SA Score, and Fsp3.
* **Novelty Verification:** Checks generated molecules against the massive PubChem database to distinguish between known compounds and truly novel discoveries.

---

## 📊 Project Workflow & Results

The NovaMol pipeline executes a complete, end-to-end drug discovery workflow, from training to validation, resulting in a ranked list of promising and novel drug candidates.

<img width="1833" height="666" alt="Screenshot 2025-10-10 142529" src="https://github.com/user-attachments/assets/2fe97c44-f5c9-4d28-83e2-1bbefa8fe6fe" />

*A sample run showing model training and real-time analysis.*

Our validation shows that the GNN can predict key properties with high accuracy, and the RNN is capable of generating a significant number of truly novel molecules.

<img width="1573" height="890" alt="Screenshot 2025-10-10 142602" src="https://github.com/user-attachments/assets/a528dfae-569c-4149-b384-5052bde19e54" />

*The final report, showing run statistics, model performance, and a table of analyzed novel molecules.*

---

## 📂 Repository Structure

```

📦 novamol-drug-discovery/
├── Home.py                  \# Main Streamlit landing page
├── SA\_Score.py              \# Local script for Synthetic Accessibility
├── pages/
│   └── 1\_🧪\_Launch\_NovaMol\_App.py \# The interactive Streamlit app
├── pipeline.py              \# Core ML pipeline for training and generation
├── requirements.txt         \# Project dependencies
└── results\_[TARGET\_ID]/     \# Output directory for models and CSVs

````

---

## ⚙️ Installation & Usage

This project is built as a Streamlit web application for an interactive, user-friendly experience.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/novamol-drug-discovery.git](https://github.com/yourusername/novamol-drug-discovery.git)
cd novamol-drug-discovery
````

### 2\. Set Up Environment

It is recommended to use a Conda environment.

```bash
# Create and activate the environment
conda create -n novamol python=3.10
conda activate novamol

# Install RDKit (required first)
conda install -c conda-forge rdkit

# Install other dependencies
pip install -r requirements.txt -f [https://data.pyg.org/whl/torch-$(python](https://data.pyg.org/whl/torch-$(python) -c "import torch; print(torch.__version__)").html
```

### 3\. Run the Application

Launch the Streamlit web server. Your browser will automatically open to the landing page.

```bash
streamlit run Home.py
```

Navigate to the "Launch NovaMol App" page from the sidebar to start using the tool.

-----

## 🔮 Future Work

  * **Expand Predictive Capabilities:** Train the GNN to predict a wider range of ADME/Tox properties (e.g., hERG inhibition, Blood-Brain Barrier penetration).
  * **Advanced Generative Models:** Explore more advanced generative architectures like Transformers or Flow-based models for molecule generation.
  * **Real-time 3D Visualization:** Integrate a 3D molecule viewer (like Py3Dmol) directly into the Streamlit app to visualize the generated candidates.

-----

## 📚 Key Technologies

  * **PyTorch & PyTorch Geometric:** For building and training the GNN and RNN models.
  * **Streamlit:** For creating the interactive web application and user interface.
  * **RDKit:** For all chemical informatics tasks, including fingerprinting and property calculation.
  * **ChEMBL & PubChem:** As primary data sources for training and validation.

<!-- end list -->

