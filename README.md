# active-learning-antiviral-polymers
A workflow to create a machine learning model for predicting antiviral activity of random copolymers. Includes steps for creating molecular descriptors of monomers and copolymers, clustering of monomers into similar groupings, training ML model and performing active learning.


This repository contains Jupyter notebooks and dependencies for developing and applying machine learning (ML) models to analyze polymer descriptors and generate large monomer libraries.

These notebooks accompany the publication:
> *[Title of your publication]* (2025).  
> *Clodagh Boland, Nathan Boase* — Macromolecular Rapid Communications.

---

## Repository Contents

| File | Description |
|------|--------------|
| [`Descriptor_ML_publication_v2.ipynb`](Descriptor_ML_publication_v2.ipynb) | Trains and evaluates ML models using molecular descriptors for polymer datasets. Uses trained models to predict antiviral activity with an active learning approach |
| [`requirements_descriptorML.txt`](requirements_descriptorML.txt) | Python dependencies required to run the Descriptor ML notebook. |
| [`Large Monomer Library Generation_publication_v3.ipynb`](Large%20Monomer%20Library%20Generation_publication_v3.ipynb) | Generates molecular descriptors for a large library of monomer structures using RDKit and related tools. Clusters monomers into groupings using UMAP and K-means clustering |
| [`requirements_largemonomerlibrary.txt`](requirements_largemonomerlibrary.txt) | Python dependencies for the Large Monomer Library Generation notebook. |

All file paths in these notebooks use **relative paths**, meaning they refer to folders *within* the cloned repository rather than absolute locations on your computer. 
This makes the workflow fully portable — as long as the folder structure is preserved, the notebooks will work on any system.

```text
active-learning-antiviral-polymers/
│
├── Descriptor_ML_publication_v2.ipynb
├── Large Monomer Library Generation_publication_v3.ipynb
│
├── data/ # Input data files (e.g., monomer lists, polymer datasets)
│ └── antiviral_dataset.csv
│
├── results/ # Outputs from the notebooks (e.g., predictions, plots)
│ └── model_outputs.csv
│
├── requirements_descriptorML.txt
└── requirements_largemonomerlibrary.txt
```
---

## Quick Start

If you use **Conda** (recommended), follow these steps to clone the repository, create an environment, and run the notebooks.

### 1. Open a terminal
On Windows, open **Anaconda Prompt**.  
On macOS or Linux, open **Terminal**.

### 2. Clone the repository
Clone this repository and navigate into it.
If need be, navigate to the directory where you want to store the project. 

```bash
git clone https://github.com/nboase/active-learning-antiviral-polymers.git
cd active-learning-antiviral-polymers

# Create and activate a new Conda environment
conda create -n antiviral-polymers python=3.11
conda activate antiviral-polymers

# Install dependencies
Depending on which notebook you want to run:
pip install -r requirements_descriptorML.txt
# or
pip install -r requirements_largemonomerlibrary.txt

# Launch Jupyter Notebook
jupyter notebook
```

Then open one of the notebooks:
Descriptor_ML_publication_v2.ipynb
or
Large Monomer Library Generation_publication_v3.ipynb

## About the Project
### Descriptor ML Notebook
1. **Data input**
Imports antiviral polymer data and performs data clean up and undersampling to normalise data distribution.
Calculates molecular descriptors for the polymers, using molecular descriptors from RDKit and Mordred generated for the monomers using 'Large Monomer Library Generation' Jupyter Notebook.
Performs feature reduction using a combination of univariate correlation analysis and recursive feature elimination (RFECV)

2. **ML Training**
Uses scikit-learn regression models (RandomForest, GradientBoosting, ElasticNet, etc.)
Hyperparameter tuning achieved using nested cross fold validation.
Outputs: model metrics (R², RMSE, MAE) and feature importance plots
Model interpretability via SHAP values

3. **Active learning**
Generates new theoretical copolymer structures and computes molecular descriptors from RDKit and Mordred generated for the monomers using 'Large Monomer Library Generation' Jupyter Notebook.
Cleans up theoretical structures based on descriptors.
Predicts antiviral activity using trained model from 2.
Enables active learning using expected improvement function.
Outputs: top 200 predicted structures from EI as .csv, visualisations of EI output.

### Large Monomer Library Notebook
Generates large sets of monomer structures from a list of CAS numbers.
Utilizes RDKit and Mordred chemistry tools to generate molecular descriptors and fingerprints for all monomers in the library
Clusters monomers into similar groupings using UMAP and K-means clustering.
Produces .sdf or .csv structure files for downstream analysis using 'Descriptor ML' notebook.

### Key Dependencies
| Library | Purpose |
|---------|---------|
| numpy, pandas | Data manipulation |
| scikit-learn | Machine learning |
| rdkit, mordred | Chemical descriptors and structure handling |
| matplotlib, seaborn, plotly |	Visualization |
| shap | Feature importance / model interpretation |

Full dependency lists are in the corresponding requirements files, which can be used to install necessary packages.

### Reproducibility
All notebooks are designed to be self-contained. You will need to use Large Monomer Library Generation before Descriptor ML notebook.
To reproduce published results, simply run the notebooks top-to-bottom using the same dependency versions listed in the requirements files.

### Citation
If you use this work, please cite:
[Your Name], [Co-authors], “[Title of Paper]”, [Journal], ([Year]).
DOI: [insert DOI or URL].

### License
This project is distributed under the GNU General Public License v3.0.
See the LICENSE file for details.

### Author
Dr Nathan Boase, 
School of Chemistry and Physics, Queensland University of Technology
nathan.boase@qut.edu.au
