#  Forecasting Seasonal Influenza Epidemics with Physics-Informed Neural Networks

This repository contains the code to evaluate and use **SIR-INN**, an epidemiological probabilistic forecasting framework based on a physics-informed neural approximation of the SIR model.

The repository is designed to be **readable and reproducible by users with no prior knowledge of Physics-Informed Neural Networks (PINNs)**. It contains the data, the model, and the code of [this work](https://arxiv.org/abs/2506.03897).

---

## Overview

The SIR-INN pipeline consists of the following conceptual steps:

0. Dataset generation and model pretraining 
1. Qualitative and quantitative evaluation of SIR-INN approximation abilities   ->   `01_SIR_approximation.ipynb`
2. SIR-INN parameters inference via MCMC                                        ->   `02_parameters_inference.ipynb`
3. Probabilistic forecasting                                                    ->   `03_forecast.ipynb`

Since the focus of this work is on model evaluation, parameter inference, and forecasting, this repository provides steps **1, 2, and 3**, for reproducibility. 
For this reason, as step **0**, we only include the synthetic dataset and the **pretrained SIR-INN model**, leaving the implementation details explanation in the work: [1].

All notebooks are intended to be executed from the repository root and can be run in Google Colab. Specifically, after cloning the repository with 

`!git clone https://github.com/martina-rama/SIR-INN.git`, 

make sure to install the required packages before running any notebook:

`!pip install -r /content/SIR-INN/requirements.txt`.

You can then execute each notebook with the following command (example for the first notebook):

```text
!jupyter nbconvert --to notebook --execute \

  --ExecutePreprocessor.kernel_name=python3 \

  --output /content/output_01_notebook.ipynb \
  
  /content/SIR-INN/notebooks/01_SIR_approximation.ipynb

The executed notebook with all outputs will be saved as `output_01_notebook.ipynb`. Figures and results will be saved in the related subfolders inside `tmp_results/`.

---

## Reference

If you use this code, please cite the following work:

[1] Rama Martina, Santin Gabriele, Cencetti Giulia, Tizzoni Michele and Lepri Bruno, *Forecasting Seasonal Influenza Epidemics with Physics-Informed Neural Networks*, arXiv preprint arXiv:2506.03897, 2025.

A BibTeX entry will be added upon publication.

---

## Repository Structure

```text
sir-inn/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── checkpoints/
│   └── SIR-INN_pretrained.pth
│
├── data/
│   └── SIR-INN_dataset.csv
│
├── src/
│   ├── models/
│   │   └── sir_inn.py
│   ├── data/
│   │   └── data_loader.py
│   ├── forecasting/
│   │   ├── inference.py
│   │   └── probabilist_forecast.py
│   ├── evaluation/
│   │   └── approximation.py
│   └── utils/
│       └── constants.py
│
└── notebooks/
    ├── 01_SIR_approximation.ipynb
    ├── 02_parameters_inference.ipynb
    └── 03_forecast.ipynb
