# Molrep-SCF

**Multimodal Molecular Representation and Property Prediction Based on Structured Complementary Feature Fusion**

Molecular representation learning plays a fundamental role in cheminformatics and computer-aided drug design (CADD), providing essential molecular features for downstream applications such as molecular property prediction, virtual screening, and drug discovery. However, obtaining sufficient high-quality experimental labels remains expensive and time-consuming, motivating the development of self-supervised learning (SSL) approaches that can exploit large-scale unlabeled molecular data.

Recent advances in multi-modal molecular representation learning have demonstrated the potential of integrating different molecular views, including SMILES sequences, 2D molecular graphs, and 3D molecular conformations. However, existing approaches still face several challenges: (1) molecular 3D structures are inherently dynamic, and representations based on a single static conformation may fail to capture conformational diversity; (2) most methods mainly focus on shared information across modalities while insufficiently modeling deep semantic interactions; and (3) existing 3D representation methods are sensitive to conformational variations, resulting in unstable geometric representations and reduced discriminative ability.

To address these challenges, we propose **Molrep-SCF**, a multi-modal self-supervised pre-training framework for molecular representation learning based on **structured complementary feature fusion**. Molrep-SCF jointly models three molecular modalities, including SMILES sequences, 2D molecular graphs, and 3D conformations, using Transformer, Graph Neural Networks (GNNs), and invariant 3D GNNs, respectively.

---

## Highlights

The main components of Molrep-SCF include:

- **Atomic-level Alignment via Position-specific Masking:**  
  We introduce a precise cross-modal alignment strategy by masking identical atomic positions across different modalities. This enables fine-grained correspondence learning among SMILES, 2D graphs, and 3D structures, improving the extraction of shared molecular semantics.

- **Structure-driven Cross-modal Semantic Coupling:**  
  We design a cross-modal reconstruction strategy that masks distinct molecular fragments and requires information recovery from other modalities. This encourages the model to capture deeper structural dependencies and complementary information beyond simple feature alignment.

- **2D-Guided Conformational Consistency Refinement:**  
  To address molecular conformational variability, we propose a 2D-guided consistency refinement mechanism that leverages molecular topology as structural prior information. Instead of relying on a single conformer representation, Molrep-SCF learns robust 3D representations by explicitly modeling consistency among multiple molecular conformations.

Extensive experiments on molecular property prediction and other downstream tasks demonstrate that Molrep-SCF achieves superior performance compared with existing multi-modal molecular representation learning methods, particularly in scenarios with limited labeled data.

---

## Model Architecture

![Molrep-SCF Framework](./modeloverview.png)
---

## Repository Structure

```text
.
├── ESPF/   # Fragment-related resources or tokenization modules used for prior-knowledge-enhanced molecular representation.
├── model/  # Model definitions and core network components.
├── process_dataset/ # Data preprocessing scripts for preparing molecular datasets and model inputs.
├── roberta-base/  # Local pretrained RoBERTa resources or configuration files used by the sequence branch. Need to obtain it from Hugging Face and add it to this project.
├── environment.yml  # Conda environment specification for reproducing the project.
├── finetune_*.py  # Entry script for downstream fine-tuning and evaluation.
├── loss.py   # Loss function definitions used in pre-training and/or fine-tuning.
├── pcqm4m.py  # Dataset loading or task-specific utilities for PCQM4M-related experiments.
├── pretrain.py  # Entry script for multi-modal pre-training.
└── utils.py  # Common helper functions.
```
## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mamengTang/Molrep-SCF.git
cd Molrep-SCF
```
### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate molrep-SCF
```
## Data Preparation

If you use **PCQM4M**, dataset-related logic may be organized in `pcqm4m.py`.
The downstream task datasets can be obtained here.
https://github.com/Hhhzj-7/MolMVC.git

## Pre-training

Run the multi-modal pre-training stage with:

```bash
python pretrain.py
```
## Fine-tuning

After pre-training, fine-tune the model on downstream molecular property prediction tasks:

```bash
python finetune_*.py
```


