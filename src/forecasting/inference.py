"""
MCMC-based rolling inference for italian influenza incidence data using the pretrained SIR-INN model.
"""

# ------------------------------------------------------------------

# Make repository importable when running as script or from notebook
import sys
from pathlib import Path
# Add repository root to Python path
repo_root = Path().resolve().parent
sys.path.append(str(repo_root))

from src.data.data_loader import load_influcast_week
from src.evaluation.approximation import incidence_from_sir, evaluation_pinn
from src.utils.constants import t0_range,epi_weeks_str,t_epi,epi_weeks,times

import torch
import functorch # required for higher-order autodiff in PINNs
import os
import pickle
import numpy as np
from tqdm import tqdm
import time
import random
from scipy.special import gammaln

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch

import pymcmcstat
from pymcmcstat.MCMC import MCMC
from pymcmcstat.propagation import define_sample_points

# ------------------------------------------------------------------

def cut_times(t_inc, y_inc, tau0_guess, forecast = False):
    """
    Align incidence times and observations after the inferred epidemic onset.

    Parameters
    ----------
    t_inc : array-like
        Discrete incidence time grid.
    y_inc : array-like or int
        Observed incidence values (or length, in forecast mode).
    tau0_guess : float
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
    t_inc = t_inc[t_inc>=tau0_guess] 
   
    if forecast:
        # Forecasting: y_inc represents only the horizon length
        min_len = np.min([len(t_inc), y_inc])

    else: 
        # Inference: ensure time grid and observations have equal length
        min_len = np.min([len(t_inc), len(y_inc)])    
        y_inc = y_inc[:min_len] 
    
    t_inc = t_inc[:min_len]
    
    return t_inc, y_inc

def model_fun_pinn(x, param, t, dt, y_inc, model):
    """
    Forward model mapping SIR-INN parameters to ILI incidence.

    Parameters
    ----------
    x : array-like
        Dummy input required by pymcmcstat (not used explicitly).
    param : array-like
        Model parameters [beta, gamma, tau0].
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

    model_eval = evaluation_pinn(param,model,t)

    # Incidence is computed on a weekly grid
    t_inc = t[0:590:dt].astype('i')

    # Align incidence with inferred epidemic onset
    t_inc, y_inc = cut_times(t_inc, y_inc, param[2])

    # Compute ILI incidence from susceptible trajectory
    val = incidence_from_sir(S=model_eval[:, 0], times=t_inc, dt=dt) 
    
    return y_inc,val

def make_ssfun_pinn(t, dt, y_inc, model):
    """
    Build a Poisson log-likelihood function for MCMC inference.
    
    This function wraps the PINN forward model and returns
    the log-likelihood under a Poisson error model,
    which is appropriate for count data.
        
    Parameters
    ----------
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
    ssfun : callable
        Sum-of-squares function (actually modified in log-likelihood) for MCMC.
    """
    
    def ssfun(theta, data, custom=None):
        y_data, fun_val = model_fun_pinn(
            data.xdata[0],
            theta,
            t,
            dt,
            y_inc,
            model
        )

        y_data=y_data*1000
        fun_val=fun_val*1000

        
        # Clip predictions to avoid log(0) or negative values
        # Add small epsilon for numerical stability
        epsilon = 1e-10
        fun_val = np.clip(fun_val, epsilon, None)
        
        # Poisson log-likelihood
        # Poisson
        #L(θ | y) = ∏ Poisson(y_i | μ_i) = ∏ (μ_i^y_i · e^(-μ_i)) / y_i!
        #Log-likelihood:
        #log L(θ | y) = Σ [y_i·log(μ_i) - μ_i - log(y_i!)]

        log_likelihood = np.sum(-fun_val + y_data * np.log(fun_val) - gammaln(y_data+1))
        
        return log_likelihood
    
    return ssfun

def run_influcast_mcmc_inference(
    season,
    country,
    dt_train,
    output_dir,
    t0_range=t0_range,
    week_labels=epi_weeks_str,
    times=times,
    pinn_model=None,
    dt=7,
    nsim=1e4,
    theta0=(0.28, 0.24, 200), 
    lb=(0.12, 1/12, 0),
    ub=(0.45, 1/2.5, 400),
    priors = 'uniform',
    seed=None
):
    """
    Run rolling-window MCMC inference on Influcast ILI data.
    For each observation time t0, a short temporal window of observations is used
    to infer epidemiological parameters (beta, gamma, tau0) leveraging either the pretrained SIR-INN or ODE solver

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
    week_labels : array-like
        Epidemiological week labels corresponding to the season.
    output_dir : str
        Directory where MCMC posterior chains are saved.
    times : array-like
        Continuous time grid used to evaluate the PINN.
    pinn_model : torch.nn.Module, optional
        Pretrained SIR-INN model (required if inference_model='pinn').    
    dt : int, optional
        Temporal resolution of incidence data (default: 7 days).
    nsim : int, optional
        Number of MCMC iterations (default: 1e4).
    theta0 : tuple, optional
        Initial guess for parameters (beta, gamma, tau0).
    lb : tuple, optional
        Lower bounds for parameters (min value of the traing set).
    ub : tuple, optional
        Upper bounds for parameters (max value of the traing set).
    priors : str, optional
        Prior distribution type for beta and gamma:
            - 'uniform': flat prior within [lb, ub].
            - 'normal': Gaussian prior centered at theta0 with sigma = (ub - lb) / 3.
        Default: 'uniform'.


    Returns
    -------
    results_dict : dict
        Dictionary containing parameter estimates over time.
    """

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Validate inputs
    if pinn_model is None:
        raise ValueError("pinn_model must be provided when inference_model='pinn'")
    
    # Create output directory
    output_subdir = f"{output_dir}/{priors}"
    os.makedirs(output_subdir, exist_ok=True)
    
    # Rolling inference over observation times
    desc = f"MCMC inference"
    for t0 in tqdm(t0_range, desc=desc):
        t_start = time.perf_counter() 

        # Select rolling observation window. Window of indices used for the inference step
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
        mcstat = MCMC(rngseed=seed)
        mcstat.data.add_data_set(x, y)

        mcstat.simulation_options.define_simulation_options(
            nsimu=int(nsim),
            updatesigma=True
        )

        # Disable pymcmcstat internal output (progress bars, tables)
        mcstat.simulation_options.waitbar = False
        mcstat.simulation_options.verbosity = 0

        if priors == 'uniform':
            # Epidemiological parameters
            mcstat.parameters.add_model_parameter(
                name="beta", theta0=theta0[0],
                minimum=lb[0], maximum=ub[0]
            )
            mcstat.parameters.add_model_parameter(
                name="gamma", theta0=theta0[1],
                minimum=lb[1], maximum=ub[1]
            )
        elif priors == 'normal':
            sigma_prior = tuple((u - l) / 3 for l, u in zip(lb, ub))

            # Epidemiological parameters
            mcstat.parameters.add_model_parameter(
                name="beta", theta0=theta0[0],
                minimum=lb[0], maximum=ub[0], prior_mu=theta0[0], prior_sigma=sigma_prior[0]
            )
            mcstat.parameters.add_model_parameter(
                name="gamma", theta0=theta0[1],
                minimum=lb[1], maximum=ub[1], prior_mu=theta0[1], prior_sigma=sigma_prior[1]
            )
        else:
            raise ValueError("Priors distribution has to be correctly specified. Please select either 'uniform' or 'normal'.")
        
        mcstat.parameters.add_model_parameter(
            name="t_0", theta0=theta0[2],
            minimum=lb[2], maximum=ub[2]
        )

        # Build sum-of-squares function (actually modified in log-likelihood)
        ssfun_local = make_ssfun_pinn(times, dt, y, pinn_model)
        
        # Define model settings
        mcstat.model_settings.define_model_settings(
            sos_function=ssfun_local
        )

        # Run MCMC
        mcstat.run_simulation()

        t_elapsed = time.perf_counter() - t_start

        results = mcstat.simulation_results.results

        # Save posterior samples
        with open(f"{output_subdir}/obs={t0}.pickle", "wb") as f:
            pickle.dump({"results": results, "runtime_seconds": t_elapsed, "seed": seed}, f)

def load_runtimes(chains_dir, t0_range, priors = 'uniform'):
    """
    Load per-t0 runtimes saved during MCMC inference.

    Parameters
    ----------
    chains_dir : str
        Base directory containing MCMC chains.
    t0_range : iterable
        Training cutoffs to load.
    priors : str, optional
        Prior type used during inference ('uniform' or 'normal'). Default: 'uniform'.

    Returns
    -------
    runtimes : dict
        {t0: runtime_seconds}
    """
    runtimes = {}
    for t0 in t0_range:
        path = f"{chains_dir}"+f"{priors}/obs={t0}.pickle"
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Support both old format (bare results) and new format (dict with runtime)
        if isinstance(data, dict) and "runtime_seconds" in data:
            runtimes[t0] = data["runtime_seconds"]
        else:
            runtimes[t0] = None  # runtime not available for this file
    return runtimes

def load_mcmc_chain(
    path,
    sample_strategy="random",
    nsample=1000,
    nsim=10000
):
    """
    Load an MCMC posterior chain and optionally subsample it.

    Parameters
    ----------
    path : str
        Path to the saved MCMC results (.pickle file).
    sample_strategy : str, optional
        Strategy used to extract samples from the chain:
        - 'tail': use samples after burn-in (i.e., last n samples of the chain)
        - 'random': random subsampling from the full chain
        - 'None': to return the full chain
    nsample : int, optional
        Number of samples to draw if sample_strategy='random', and to select at the end of the chain if sample_strategy='tail'.
    nsim : int, optional
        Total number of MCMC iterations (used for random subsampling).

    Returns
    -------
    chain : ndarray
        Array of sampled parameter vectors with shape (n_samples, n_params).
    """
    
    # Load MCMC results dictionary
    with open(path, "rb") as f:
        file_results = pickle.load(f)
        results = file_results["results"]

    # Full posterior chain (nsim x n_params)
    chain = results["chain"]

    # Use only the tail of the chain (after burn-in)
    if sample_strategy == "tail":
        chain = chain[-nsample:]

    # Randomly subsample posterior draws
    elif sample_strategy == "random":
        idx_sample, _ = pymcmcstat.propagation.define_sample_points(
            nsample=nsample,
            nsimu=nsim
        )
        chain = chain[idx_sample]

    elif sample_strategy == "None":
        chain = chain        

    return chain

def save_mcmc_chain_tails(
    path,
    dt_train,
    times=times,
    t0_range=t0_range,
    pinn_model=None,
    sample_strategy="tail",
    nsample=1000,
    nsim=10000
):

    """
    Load and aggregate MCMC posterior samples across all inference time steps.
    
    For each observation cutoff t0, loads the saved posterior chain,
    extracts parameter samples (beta, gamma, tau0, R0), and computes
    the effective reproduction number Rt = R0 * S(t) at the current
    week using the pretrained SIR-INN. Results are serialized to a
    single .pkl file for downstream use in forecasting.
    
    Parameters
    ----------
    path : str
        Directory containing per-t0 MCMC pickle files and where
        the output mcmc_distributions.pkl will be saved.
    times : array-like
        Continuous time grid used to evaluate the PINN.
    t0_range : iterable
        Observation cutoff indices over which inference was performed.
    dt_train : int
        Length of the rolling observation window (number of weeks).
    pinn_model : torch.nn.Module
        Pretrained SIR-INN model used to compute S(t) for Rt estimation.
    sample_strategy : str, optional
        Subsampling strategy passed to load_mcmc_chain
        ('tail', 'random', or 'None'). Default: 'tail'.
    nsample : int, optional
        Number of samples to extract per chain. Default: 1000.
    nsim : int, optional
        Total MCMC iterations per chain (used for random subsampling). Default: 10000.
    
    Returns
    -------
    None
        Saves mcmc_distributions.pkl to path containing:
        - 'weeks'          : array of t0 indices
        - 'betas_samples'  : list of beta posterior arrays
        - 'gammas_samples' : list of gamma posterior arrays
        - 'tau0s_samples'  : list of tau0 posterior arrays
        - 'R0s_samples'    : list of R0 = beta/gamma posterior arrays
        - 'Rts_samples'    : list of Rt = R0 * S(t) posterior arrays
    """
        
    betas_samples = []
    gammas_samples = []
    tau0s_samples = []
    R0s_samples = []
    Rts_samples = []
    weeks = []

    t_inc = times[0:590:7].astype('i')

    for t0_train in t0_range:
        # Load MCMC posterior samples after burnin
        chain_path = path+f"obs={t0_train}.pickle"
        chain = load_mcmc_chain(
            chain_path,
            sample_strategy=sample_strategy,
            nsample=nsample,
            nsim=nsim
        )

        betas = chain[:, 0]
        gammas = chain[:, 1]
        tau0s = chain[:, 2]
        R0s = betas / gammas

        # Store ALL samples for each parameter 
        betas_samples.append(betas)
        gammas_samples.append(gammas)
        tau0s_samples.append(tau0s)
        R0s_samples.append(R0s)
        weeks.append(t0_train)

        # Rt posterior
        Rt_week_samples = []

        # Identify time window of observations (used to truncate forecasts)
        window_train = np.arange(t0_train - dt_train + 1, t0_train + 1)
        window_train = window_train[window_train >= 0]

        # Compute Rt = R0*S
        for beta_j, gamma_j, tau0_j, R0_j in zip(betas, gammas, tau0s, R0s):
        
            # Cut time axis based on epidemic onset t0
            t0_j_inc, y_inc = cut_times(t_inc, len(window_train), tau0_j, forecast=True)
            t0_j_current = t0_j_inc[-1]

            params_epi = np.array([beta_j, gamma_j])
            model_eval = evaluation_pinn(params_epi, pinn_model, np.array([t0_j_current]))
            S_at_t0 = model_eval[:, 0][0]

            Rt_week_samples.append(R0_j * S_at_t0)

        Rts_samples.append(np.array(Rt_week_samples))

        data_dict = {
            "weeks": np.array(weeks),
            "betas_samples": betas_samples,
            "gammas_samples": gammas_samples,
            "tau0s_samples": tau0s_samples,
            "R0s_samples": R0s_samples,
            "Rts_samples": Rts_samples
        }

    output_path = os.path.join(path, "mcmc_distributions.pkl")

    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"File saved in: {output_path}")
    
def plot_inferred_parameters(
    chains_dir,
    t_epi=t_epi,
    epi_weeks=epi_weeks
):
    
    """
    Plot posterior distributions of inferred epidemiological parameters over time.
    
    Loads aggregated MCMC samples from mcmc_distributions.pkl and produces
    a 2x2 panel of boxplots showing the weekly posterior distributions of
    beta, gamma, tau0, and Rt. Each panel includes the overall posterior
    median as a reference line.
    
    Parameters
    ----------
    t_epi : array-like
        Continuous ordered time index (one entry per epidemiological week),
        used as x-axis positions for the boxplots.
    epi_weeks : array-like
        Epidemiological week labels (e.g. [42, 43, ..., 52, 1, ..., 17]),
        used as x-axis tick labels.
    chains_dir : str
        Directory containing the mcmc_distributions.pkl file.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the 2x2 parameter panel.
    """

    with open(chains_dir + "mcmc_distributions.pkl", "rb") as f:
        data = pickle.load(f)
  
    weeks_ordered = t_epi
    weeks_labels = epi_weeks
    
    # Colori
    box_color = "#2A6F9E"      
    median_color = "#D55E00" 
    global_median_color = "#333333"
        
    weeks = data["weeks"]           
    betas_samples = data["betas_samples"]  
    gammas_samples= data["gammas_samples"]
    tau0s_samples = data["tau0s_samples"]
    Rts_samples = data["Rts_samples"]
    
    # Compute global median
    betas_global = np.median(np.concatenate(betas_samples))
    gammas_global = np.median(np.concatenate(gammas_samples))
    tau0s_global = np.median(np.concatenate(tau0s_samples))
    Rts_global = np.median(np.concatenate(Rts_samples))
    
    # Create figure
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        dpi=300,
        #sharex=True
    )
    axes = axes.flatten()
    
    # Parameter configurations: samples, global median, label, y-limits
    params = [
        (betas_samples, betas_global, r'$\hat{\beta}$', (0, 1)),
        (gammas_samples, gammas_global, r'$\hat{\gamma}$', (0, 1)),
        (tau0s_samples, tau0s_global, r'$\hat{\tau}_0$', (0, 600)),
        (Rts_samples, Rts_global, r'$\hat{R}_t$', (0.5, 1.5))
    ]
    
    for i, (ax, (samples_list, median_val, title, ylim)) in enumerate(zip(axes, params)):
            
        # Special case for Rt
        if title == r'$\hat{R}_t$':
            ax.axhline(
                y=1,
                linestyle='-',
                lw=1.8,
                color=global_median_color,
                alpha=0.9,
                label=r"$R_t = 1$",
                zorder=1
            )
        else:
            ax.axhline(
                y=median_val,
                linestyle=(0, (4, 4)),
                lw=2,
                color=global_median_color,
                alpha=0.9,
                label=rf"Overall median = {median_val:.2f}",
                zorder=1
            )
        
        # Create boxplot
        positions = weeks_ordered[3:]  # Starting from week 3
        
        bp = ax.boxplot(
            samples_list,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,  # Hide outliers for cleaner look
            boxprops=dict(facecolor=box_color, alpha=0.6, edgecolor=box_color, linewidth=1.5),
            whiskerprops=dict(color=box_color, linewidth=1.5),
            capprops=dict(color=box_color, linewidth=1.5),
            medianprops=dict(color=median_color, linewidth=2.5),
            zorder=2
        )
            
        if title == r'$\hat{R}_t$':
            first_legend = plt.Line2D(
                [0], [0],
                color=global_median_color,
                linestyle='-',
                lw=1.8,
                label=r"$R_t = 1$"
            )
        else:
            first_legend = plt.Line2D(
                [0], [0],
                color=global_median_color,
                linestyle=(0, (4, 4)),
                lw=2,
                label=rf"Overall median = {median_val:.2f}"
            )
    
        legend_elements = [
            first_legend,
            Patch(facecolor=box_color, alpha=0.6, edgecolor=box_color,
                  label='Posterior distribution (IQR)'),
            plt.Line2D([0], [0], color=median_color, lw=2.5, label='Median')
        ]
        
        # Set title and labels
        ax.set_title(title, fontsize=17, fontweight='bold', pad=12)
        
        if i in [0, 2]:
            ax.set_ylabel("Parameter value", fontsize=12, labelpad=12)
        
        if i in [2, 3]:
            ax.set_xlabel("Week", fontsize=12, labelpad=12)
        
        ax.set_ylim(*ylim)
    
        # Epidemiological week formatting
        ax.set_xticks(weeks_ordered)
        ax.set_xticklabels(weeks_labels)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis='x', which='major', labelbottom=True)  # forza la visibilità
        
        # Styling
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=11)
        ax.legend(handles=legend_elements, fontsize=10, loc='best')
    
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.5)
    
    return fig

def plot_inferred_R0(
    chains_dir,
    t_epi=t_epi,
    epi_weeks=epi_weeks
):

    """
    Plot the posterior distribution of the basic reproduction number R0 over time.
    
    Loads aggregated MCMC samples from mcmc_distributions.pkl and produces
    a single boxplot panel showing the weekly posterior distribution of
    R0 = beta / gamma, together with the overall posterior median.
    
    Parameters
    ----------
    t_epi : array-like
        Continuous ordered time index (one entry per epidemiological week),
        used as x-axis positions for the boxplots.
    epi_weeks : array-like
        Epidemiological week labels (e.g. [42, 43, ..., 52, 1, ..., 17]),
        used as x-axis tick labels.
    chains_dir : str
        Directory containing the mcmc_distributions.pkl file.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the R0 boxplot panel.
    """

    with open(chains_dir + "mcmc_distributions.pkl", "rb") as f:
        data = pickle.load(f)
  
    weeks_ordered = t_epi
    weeks_labels = epi_weeks
    
    box_color = "#2A6F9E"    
    median_color = "#D55E00" 
    global_median_color = "#333333"

    R0s_samples = data["R0s_samples"]
    
    # Compute global median
    R0s_global = np.median(np.concatenate(R0s_samples))
    
    # Create figure
    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 5),
        dpi=300
    )
    
    # Parameter configurations: samples, global median, label, y-limits
    params = [
        (R0s_samples, R0s_global, r'$\hat{R}_0$', (1, 2))
    ]
    
    for i, (samples_list, median_val, title, ylim) in enumerate(params):
        
        # Plot global posterior mean (horizontal line)
        ax.axhline(
            y=median_val,
            #linestyle='--',
            linestyle=(0, (4, 4)),
            lw=2,
            color=global_median_color,
            alpha=0.9,
            label=rf"Overall median = {median_val:.2f}",
            zorder=1
        )
        
        # Create boxplot
        positions = weeks_ordered[3:]  # Starting from week 3
        
        bp = ax.boxplot(
            samples_list,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,  # Hide outliers for cleaner look
            boxprops=dict(facecolor=box_color, alpha=0.6, edgecolor=box_color, linewidth=1.5),
            whiskerprops=dict(color=box_color, linewidth=1.5),
            capprops=dict(color=box_color, linewidth=1.5),
            medianprops=dict(color=median_color, linewidth=2.5),
            zorder=2
        )
        
        # Add custom legend entry for boxplot
        legend_elements = [
            plt.Line2D(
                [0], [0],
                color=global_median_color,
                #linestyle='--',
                linestyle=(0, (4, 4)),
                lw=2,
                label=rf"Overall median = {median_val:.2f}"
            ),
            Patch(facecolor=box_color, alpha=0.6, edgecolor=box_color,
                  label='Posterior distribution (IQR)'),
            plt.Line2D([0], [0], color=median_color, lw=2.5, label='Median')
        ]
        
        # Set title and labels
        ax.set_title(title, fontsize=17, fontweight='bold', pad=12)
        
        ax.set_ylabel("Parameter value", fontsize=12, labelpad=12)
        
        ax.set_xlabel("Week", fontsize=12, labelpad=12)
        
        ax.set_ylim(*ylim)
        
        # Epidemiological week formatting
        ax.set_xticks(weeks_ordered)
        ax.set_xticklabels(weeks_labels)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        
        # Styling
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=11)
        ax.legend(handles=legend_elements, fontsize=10, loc='best')
    
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.5)
    
    return fig

def load_season_medians(chains_dir):
    
    with open(os.path.join(chains_dir, "mcmc_distributions.pkl"), "rb") as f:
        data = pickle.load(f)

    betas_samples = data["betas_samples"]
    gammas_samples = data["gammas_samples"]

    betas_median = np.array([np.median(b) for b in betas_samples])
    gammas_median = np.array([np.median(g) for g in gammas_samples])

    return betas_median, gammas_median

def plot_training_set_with_parameters(x_train_np, idx_train, results_dir, priors='uniform'):

    """
    Scatter plot comparing the SIR-INN training grid with inferred parameter pairs.
    
    Overlays the (beta, gamma) pairs from the training dataset with the
    weekly posterior medians inferred for the 2023-2024 and 2024-2025
    influenza seasons, allowing visual assessment of whether real-season
    parameters fall within the training distribution.
    
    Parameters
    ----------
    x_train_np : np.ndarray
        Training input array of shape (N, 3) with columns [time, beta, gamma].
    idx_train : np.ndarray
        Scenario indices identifying each trajectory in the training set,
        used to extract one (beta, gamma) pair per scenario.
    results_dir : str
        Base directory containing per-season inference results, expected to
        have subfolders of the form {season}/{priors}/.
    priors : str, optional
        Prior type used during inference ('uniform' or 'normal'). Default: 'uniform'.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the scatter plot.
    """
    
    color_2324 = "#1f4e79"   # blu scuro
    color_2425 = "#cc2c7a"   # magenta

    betas_train = []
    gammas_train = []

    for idx in np.unique(idx_train):
        i0 = np.where(idx_train == idx)[0][0]
        betas_train.append(x_train_np[i0, 1])
        gammas_train.append(x_train_np[i0, 2])

    betas_train = np.array(betas_train)
    gammas_train = np.array(gammas_train)

    fig, ax = plt.subplots(figsize=(7,7))
    
    # Training grid
    plt.scatter(
        betas_train, gammas_train,
        s=15, alpha=0.25,
        label="Training set"
    )

    # 2023-2024
    betas_2324, gammas_2324 = load_season_medians(results_dir+f"2023-2024/{priors}/")

    # 2024-2025
    betas_2425, gammas_2425 = load_season_medians(results_dir+f"2024-2025/{priors}/")

    # Season 2023-2024
    ax.scatter(
        betas_2324, gammas_2324,
        s=70, marker="o",
        color=color_2324,
        edgecolor="black",
        label="2023-2024 weekly medians"
    )

    # Season 2024-2025
    ax.scatter(
        betas_2425, gammas_2425,
        s=70, marker="o",
        color=color_2425,
        edgecolor="black",
        label="2024-2025 weekly medians"
    )

    ax.set_xlabel(r'$\beta$', fontsize=15, labelpad = 7)
    ax.set_ylabel(r'$\gamma$', fontsize=15, labelpad = 7)
    ax.set_title("Training vs inferred parameter pairs", fontsize=16, pad = '15')

    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=12, loc='best') #handles=legend_elements


    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig