# SIR-INN: Physics-Informed Neural Networks for Influenza Forecasting

This repository contains the code to evaluate and use **SIR-INN**, a continuous-time epidemiological forecasting framework based on a physics-informed neural approximation of the SIR model.

The repository is designed to be **readable and reproducible by users with no prior knowledge of Physics-Informed Neural Networks (PINNs)**.
For this reason, training and dataset generation are encapsulated, and the focus is on model evaluation, parameter inference, and forecasting.

---

## Overview

The SIR-INN pipeline consists of the following conceptual steps:

1. Dataset generation and model pretraining *(not exposed)*
2. Final training of the SIR-INN model *(encapsulated)*
3. Qualitative evaluation of the reconstructed SIR dynamics
4. Projection of latent dynamics to observable incidence curves
5. Parameter inference and probabilistic forecasting
6. Export of forecasts in Influcast Hub–compatible format

This repository exposes steps **3, 5, and 6**, while providing a **pretrained SIR-INN model** for reproducibility. The code prioritizes epidemiological interpretability and reproducibility over implementation details related to physics-informed learning.

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
├── src/
│   ├── models/
│   │   └── sir_inn.py
│   ├── evaluation/
│   │   ├── incidence.py
│   │   ├── metrics.py
│   │   ├── postprocessing.py
│   │   └── export_influcast.py
│   └── utils/
│
├── notebooks/
│   ├── 01_qualitative_fit_and_incidence.ipynb
│   ├── 02_forecast_mcmc.ipynb
│   └── 03_save_forecasts.ipynb
│
└── experiments/
    └── run_forecast_mcmc.py

----

## Reference

If you use this code, please cite the following work:

> Rama Martina, Santin Gabriele, Cencetti Giulia, Tizzoni Michele and Lepri Bruno, *Forecasting Seasonal Influenza Epidemics with Physics-Informed Neural Networks*,  
> arXiv preprint arXiv:2506.03897, 2025

A BibTeX entry will be added upon publication.
