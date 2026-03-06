# Influence Mapping

Code for the analysis and modeling presented in the preprint:

**Principles of cortical interactions in modular recurrent networks**  
https://www.biorxiv.org/content/10.64898/2025.12.19.695176v2

---

# Requirements

## Operating System

The code was tested on:

- **macOS Tahoe 26.3 (25D125)**

## Python Version

- **Python 3.13.1**
- **pip 24.2**

## Python Dependencies

Required Python packages:
```
dataset==1.6.2
h5py==3.14.0
matplotlib==3.10.8
numpy==2.4.2
pandas==3.0.1
psutil==5.9.0
scipy==1.17.1
statsmodels==0.14.4
torch==2.6.0
tqdm==4.67.1
```

---

# Installation

## 1. Clone the Repository
```
git clone https://github.com/kaschube-lab/Influence_mapping.git
cd Influence_mapping
```
## 2. Install Python Dependencies

```
pip install -r requirements.txt
```
Typical installation time: **less than 1 minute** on a standard 8-core laptop.

---

# Demo

A demonstration of the core simulation is provided in:

```
./demos/fig4.ipynb
```
Run the notebook using **Jupyter Notebook** or **JupyterLab**.

The demo reproduces the main simulation results shown in **Figure 4** of the paper.

Expected runtime: **~5 minutes on an 8-core laptop**.

---

# Instructions for Use

To run simulations with your own parameters, see the examples in

```
./demos/fig4.ipynb
```

The notebook demonstrates:
- how to configure the model parameters
- how to run the network simulations



