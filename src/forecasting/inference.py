"""
MCMC-based rolling inference for influenza incidence data
using the pretrained SIR-INN model.
"""

# ------------------------------------------------------------------

# Make repository importable when running as script or from notebook
import sys
from pathlib import Path
# Add repository root to Python path
repo_root = Path().resolve().parent
sys.path.append(str(repo_root))

from src.data.data_loader import load_influcast_week
from src.evaluation.epidemiology import incidence_from_sir

import torch
import functorch # required for higher-order autodiff in PINNs
import os
import pickle
import numpy as np
from tqdm import tqdm

from pymcmcstat.MCMC import MCMC

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ------------------------------------------------------------------

def cut_times(t_inc, y_inc, t0_guess, forecast = False):
    """
    Align incidence times and observations after the inferred epidemic onset.

    Parameters
    ----------
    t_inc : array-like
        Discrete incidence time grid.
    y_inc : array-like or int
        Observed incidence values (or length, in forecast mode).
    t0_guess : float
        Estimated epidemic start time.
    forecast : bool, optional
        If True, only the time grid is truncated (used during forecasting).

    Returns
    -------
    t_inc : array
        Truncated incidence time grid.
    y_inc : array
        Truncated incidence observations (in inference mode).
    """

    # Keep only incidence times after the inferred epidemic onset
    t_inc = t_inc[t_inc>=t0_guess] 
   
    if forecast:
        # Forecasting: y_inc represents only the horizon length
        min_len = np.min([len(t_inc), y_inc])

    else: 
        # Inference: ensure time grid and observations have equal length
        min_len = np.min([len(t_inc), len(y_inc)])    
        y_inc = y_inc[:min_len] 
    
    t_inc = t_inc[:min_len]
    
    return t_inc, y_inc

def model_fun0(x, param, t, dt, y_inc, model):
    """
    Forward model mapping SIR-INN parameters to ILI incidence.

    Parameters
    ----------
    x : array-like
        Dummy input required by pymcmcstat (not used explicitly).
    param : array-like
        Model parameters [beta, gamma, t0].
    t : array-like
        Continuous time grid.
    dt : int
        Time discretization step.
    y_inc : array-like
        Observed incidence data.
    model : torch.nn.Module
        Pretrained SIR-INN model.

    Returns
    -------
    y_inc : array
        Observed incidence values.
    val : array
        Model-predicted incidence values.
    """

    # Number of evaluation points
    n_times = len(t)

    # Build PINN input: time + replicated epidemiological parameters
    input_eval = torch.tensor(np.hstack([t[:, None], np.tile(param[:2], (n_times, 1))])).float()

    # Evaluate SIR-INN without gradient tracking
    with torch.no_grad():
        model_eval = model(input_eval).numpy()

    # Incidence is computed on a weekly grid
    t_inc = t[0:590:dt].astype('i')

    # Align incidence with inferred epidemic onset
    t_inc, y_inc = cut_times(t_inc, y_inc, param[2])

    # Compute ILI incidence from susceptible trajectory
    val = incidence_from_sir(S=model_eval[:, 0], times=t_inc, dt=dt) 
    
    return y_inc,val

def make_ssfun(t, dt, y_inc, model):
    """
    Build a sum-of-squares function for MCMC inference.

    This function wraps the PINN forward model and returns
    the normalized squared error between observed and predicted incidence.
    """
    
    def ssfun(theta, data, custom=None):
        y_data, fun_val = model_fun0(
            data.xdata[0],
            theta,
            t,
            dt,
            y_inc,
            model
        )
        res = y_data - fun_val
        return (res**2).sum() / len(res)
    return ssfun

def run_influcast_mcmc_inference(
    season,
    country,
    t0_range,
    dt_train,
    pinn_model,
    week_labels,
    output_dir,
    times,
    dt=7,
    nsimu=1e4,
    theta0=(0.2, 0.15, 200),
    lb=(0.12, 1/12, 0),
    ub=(0.45, 1/2.5, 400),
):
    """
    Run rolling-window MCMC inference on Influcast ILI data.
    For each observation time t0, a short temporal window of observations is used
    to infer epidemiological parameters (beta, gamma, t0) leveraging the pretrained SIR-INN.

     Parameters
    ----------
    season : str
        Influenza season identifier (e.g. '2023-2024').
    country : str
        Country name used to load surveillance data (e.g. 'italia').
    t0_range : iterable
        Indices of the observation times at which inference is performed.
    dt_train : int
        Length of the rolling training window (number of weeks).
    pinn_model : torch.nn.Module
        Pretrained SIR-INN model.
    week_labels : array-like
        Epidemiological week labels corresponding to the season.
    output_dir : str
        Directory where MCMC posterior chains are saved.
    times : array-like
        Continuous time grid used to evaluate the PINN.
    dt : int, optional
        Temporal resolution of incidence data (default: 7 days).
    nsimu : int, optional
        Number of MCMC iterations (default: 1e4).
    theta0 : tuple, optional
        Initial guess for parameters (beta, gamma, t0).
    lb : tuple, optional
        Lower bounds for parameters (min value of the traing set).
    ub : tuple, optional
        Upper bounds for parameters (max value of the traing set).
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Rolling inference over observation times
    for t0 in tqdm(t0_range, desc="MCMC rolling inference"):
        # Select rolling observation window. Window of indices used for training at current t0
        window = np.arange(t0 - dt_train + 1, t0 + 1)
        window = window[window >= 0]

        # Dummy x-data required by pymcmcstat
        x = window.reshape(len(window), 1)

        # Identify current epidemiological week
        week_id = week_labels[window][-1]

        # Load observed incidence up to current week
        incidence = load_influcast_week(
            season=season,
            country=country,
            week=week_id
        )[-dt_train:]

        y = np.asarray(incidence)

        # Initialize MCMC
        mcstat = MCMC()
        mcstat.data.add_data_set(x, y)

        mcstat.simulation_options.define_simulation_options(
            nsimu=int(nsimu),
            updatesigma=True
        )

        # Disable pymcmcstat internal output (progress bars, tables)
        mcstat.simulation_options.waitbar = False
        mcstat.simulation_options.verbosity = 0

        # Epidemiological parameters
        mcstat.parameters.add_model_parameter(
            name="beta", theta0=theta0[0],
            minimum=lb[0], maximum=ub[0]
        )
        mcstat.parameters.add_model_parameter(
            name="gamma", theta0=theta0[1],
            minimum=lb[1], maximum=ub[1]
        )
        mcstat.parameters.add_model_parameter(
            name="t_0", theta0=theta0[2],
            minimum=lb[2], maximum=ub[2]
        )

        # Build sum-of-squares function
        t, dt, y_inc, model = times, dt, y, pinn_model
        ssfun_local = make_ssfun(t, dt, y_inc, model)

        # Model wrapper (already defined elsewhere)
        mcstat.model_settings.define_model_settings(
            sos_function=ssfun_local
        )

        # Run MCMC
        mcstat.run_simulation()

        results = mcstat.simulation_results.results

        # Save posterior samples
        with open(f"{output_dir}/obs={t0}.pickle", "wb") as f:
            pickle.dump(results, f)

def load_mcmc_parameter_estimates(
    t0_range,
    chains_dir,
    burnin=5000
):
    """
    Load MCMC chains and compute posterior means for each parameter, for each rolling inference step.

    Parameters
    ----------
    t0_range : iterable
        Indices of observation times corresponding to saved chains.
    chains_dir : str
        Directory containing MCMC result files.
    burnin : int, optional
        Number of initial MCMC samples to discard (default: 5000).

    Returns
    -------
    betas : list
        Posterior mean estimates of beta over time.
    gammas : list
        Posterior mean estimates of gamma over time.
    t0s : list
        Posterior mean estimates of epidemic onset times.
    R0s : list
        Posterior mean estimates of the basic reproduction number R0, computed as beta/gamma.
    """

    betas, gammas, t0s, R0s = [], [], [], []

    for t0 in tqdm(t0_range, desc="Loading MCMC chains"):
        # Load MCMC results
        with open(f"{chains_dir}/obs={t0}.pickle", "rb") as f:
            results = pickle.load(f)

        # Chain shape: (nsimu, n_parameters)
        chain = results["chain"]  

        # Compute posterior mean after burn-in
        post_mean = np.mean(chain[burnin:], axis=0)
        beta_hat, gamma_hat, t0_hat = post_mean

        # Store estimates
        betas.append(beta_hat)
        gammas.append(gamma_hat)
        t0s.append(t0_hat)
        R0s.append(beta_hat / gamma_hat)

    return betas, gammas, t0s, R0s


def plot_mcmc_parameters(
    weeks_ordered,
    weeks_labels,
    betas, gammas, t0s, R0s,
    betas_mean, gammas_mean, t0s_mean, R0s_mean
):
    """
    Visualize the temporal evolution of inferred epidemiological parameters.
    
    Parameters
    ----------
    weeks_ordered : array-like
        Numerical indices of epidemiological weeks.
    weeks_labels : array-like
        String labels for epidemiological weeks.
    betas, gammas, t0s, R0s : list
        Time series of posterior mean parameter estimates.
    betas_mean, gammas_mean, t0s_mean, R0s_mean : float
        Global mean estimates over the season.
    """
    
    fig, axes = plt.subplots(1, 4, figsize=(13, 5), dpi=300)

    # Parameter configurations: values, global mean, label, y-limits
    params = [
        (betas, betas_mean, r'$\hat{\beta}$', (0, 1)),
        (gammas, gammas_mean, r'$\hat{\gamma}$', (0, 1)),
        (t0s, t0s_mean, r'$\hat{t}_0$', (0, 600)),
        (R0s, R0s_mean, r'$\hat{R}_0$', (1, 2)),
    ]

    for i, (ax, (values, mean_val, title, ylim)) in enumerate(zip(axes, params)):
        # Plot global posterior mean
        ax.axhline(
            y=mean_val,
            linestyle='--',
            lw=2,
            color='palevioletred',
            label=rf"Average = {mean_val:.2f}"
        )

        # Plot rolling MCMC estimates
        ax.plot(
            weeks_ordered[3:], values,
            '.-', color='mediumvioletred',
            markersize=8,
            alpha=0.8,
            label="MCMC estimate"
        )

        # Label y-axis only once for clarity
        if i == 0:
            ax.set_ylabel("Parameters behavior", fontsize=12)

        ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel("Week", fontsize=12)
        ax.set_ylim(*ylim)

        # Epidemiological week formatting
        ax.set_xticks(weeks_ordered)
        ax.set_xticklabels(weeks_labels)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        # Styling
        ax.grid(True, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=11)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()