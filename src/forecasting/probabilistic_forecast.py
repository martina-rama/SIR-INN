"""
Probabilistic forecasting utilities based on MCMC posterior samples
and the pretrained SIR-INN model.

This module handles:
- posterior chain loading and subsampling,
- forward simulation of incidence trajectories,
- aggregation into prediction intervals and quantiles,
- visualization of probabilistic forecasts.
"""

# ------------------------------------------------------------------

# Make repository importable when running as script or from notebook
import sys
from pathlib import Path
# Add repository root to Python path
repo_root = Path().resolve().parent
sys.path.append(str(repo_root))

from src.evaluation.epidemiology import incidence_from_sir
from src.forecasting.inference import cut_times
from src.data.data_loader import load_influenza_season

import torch
import functorch # required for higher-order autodiff in PINNs
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

import pymcmcstat
from pymcmcstat.propagation import define_sample_points

# ------------------------------------------------------------------
# Define the quantiles of interest for the probabilistic forecasts
quantiles = [0.01, 0.02, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25,
             0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75,
             0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
rename_map = {
    0.01: 'q01',
    0.02: 'q02',
    0.025: 'q025',
    0.05: 'q05',
    0.10: 'q10',
    0.15: 'q15',
    0.20: 'q20',
    0.25: 'q25',
    0.30: 'q30',
    0.35: 'q35',
    0.40: 'q40',
    0.50: 'q50',
    0.60: 'q60',
    0.65: 'q65',
    0.70: 'q70',
    0.75: 'q75',
    0.80: 'q80',
    0.85: 'q85',
    0.90: 'q90',
    0.95: 'q95',
    0.975: 'q975',
    0.99: 'q99'
}
# ------------------------------------------------------------------

def load_mcmc_chain(
    path,
    sample_strategy="random",
    burnin=500,
    nsample=500,
    nsimu=10000
):
    """
    Load an MCMC posterior chain and optionally subsample it.

    Parameters
    ----------
    path : str
        Path to the saved MCMC results (.pickle file).
    sample_strategy : str, optional
        Strategy used to extract samples from the chain:
        - 'tail': use samples after burn-in
        - 'random': random subsampling from the full chain
    burnin : int, optional
        Number of initial samples to discard if sample_strategy='tail'.
    nsample : int, optional
        Number of samples to draw if sample_strategy='random'.
    nsimu : int, optional
        Total number of MCMC iterations (used for random subsampling).

    Returns
    -------
    chain : ndarray
        Array of sampled parameter vectors with shape (n_samples, n_params).
    """
    
    # Load MCMC results dictionary
    with open(path, "rb") as f:
        results = pickle.load(f)

    # Full posterior chain (nsimu x n_params)
    chain = results["chain"]

    # Use only the tail of the chain (after burn-in)
    if sample_strategy == "tail":
        chain = chain[burnin:]

    # Randomly subsample posterior draws
    elif sample_strategy == "random":
        idx_sample, _ = pymcmcstat.propagation.define_sample_points(
            nsample=nsample,
            nsimu=nsimu
        )
        chain = chain[idx_sample]

    return chain

def forecast_pinn(x, model, times, dt, window):
    """
    Generate an incidence forecast from the PINN for a single parameter set.

    Parameters
    ----------
    x : array-like
        Epidemiological parameters [beta, gamma, t0].
    model : torch.nn.Module
        Pretrained SIR-INN model.
    times : array-like
        Continuous time grid used for PINN evaluation.
    dt : int
        Time discretization used for incidence extraction.
    window : int
        Length of the forecast window.

    Returns
    -------
    incidence_estimated : ndarray
        Predicted incidence trajectory.
    """

    # Build PINN input: [time, beta, gamma]
    input_eval = torch.tensor(np.hstack([times[:, None], np.tile([x[0], x[1]], (len(times), 1))])).float()

    # Forward evaluation of the PINN
    with torch.no_grad():
        model_eval = model(input_eval).numpy()

    # Discrete times at which incidence is computed
    t_inc = times[0:590:dt].astype('i')

    # Cut time axis based on epidemic onset t0
    t_inc,y_inc = cut_times(t_inc, window, x[2], forecast = True)

    # Compute incidence from susceptible trajectory
    incidence_estimated = incidence_from_sir(S=model_eval[:, 0], times=t_inc, dt=dt)
    
    return incidence_estimated

def forecast_TV(time_plot_x,chain_samples,model,times,dt,window_forecast):
    """
    Generate stacked forecast trajectories over multiple posterior samples.

    Parameters
    ----------
    time_plot_x : array-like
        Time indices used for plotting.
    chain_samples : ndarray
        Sampled parameter vectors from the posterior.
    model : torch.nn.Module
        Pretrained SIR-INN model.
    times : array-like
        Continuous time grid.
    dt : int
        Time discretization.
    window_forecast : array-like
        Forecast window indices.

    Returns
    -------
    T : ndarray
        Stacked time indices.
    V : ndarray
        Stacked incidence values.
    """
    
    forecast = {'pinn': [], 't': []}

    # First sample initializes arrays
    forecast['t'] = time_plot_x
    forecast['pinn'] = forecast_pinn(chain_samples[0], model, times, dt, len(window_forecast))

    # Loop over remaining posterior samples
    for params in chain_samples[1:]: # [::10]
        forecast['t'] = np.concatenate((forecast['t'], time_plot_x), axis=0)
        forecast['pinn'] = np.concatenate((forecast['pinn'], forecast_pinn(params, model, times, dt, len(window_forecast))), axis=0)

    # Flatten for DataFrame construction
    T = np.vstack(forecast['t'])[:, 0]
    V = np.vstack(forecast['pinn'])[:, 0]
    
    return T,V

def generate_forecast_samples(
    t0_train,
    dt_train,
    dt_forecast,
    chain_samples,
    pinn_model,
    times,
    dt,
    season_offset=42,
    season_length=28
):
    """
    Generate forecast samples for a single cutoff t0.

    Parameters
    ----------
    t0_train : int
        Index of the last observed time.
    dt_train : int
        Length of the training window.
    dt_forecast : int
        Forecast horizon.
    chain_samples : ndarray
        Posterior samples.
    pinn_model : torch.nn.Module
        Pretrained PINN.
    times : array-like
        Continuous time grid.
    dt : int
        Time discretization.
    season_offset : int, optional
        Starting epidemiological week of the season.
    season_length : int, optional
        Number of weeks in the season.

    Returns
    -------
    T, V : ndarray
        Stacked forecast times and values.
    window_forecast : ndarray
        Epidemiological weeks corresponding to the forecast.
    """

    # Define forecast window in seasonal coordinates
    window_forecast = np.arange(
        t0_train - dt_train + 1,
        t0_train + 1 - dt_train + dt_forecast
    )
    window_forecast = season_offset + window_forecast[
        (window_forecast >= 0) & (window_forecast < season_length)
    ]

    # Convert to plotting index
    time_plot_x = window_forecast - (season_offset - 1)

    # Generate forecast trajectories
    T, V = forecast_TV(
        time_plot_x,
        chain_samples,
        pinn_model,
        times,
        dt,
        window_forecast
    )

    return T, V, window_forecast

def truncate_forecast_blocks(
    T,
    V,
    window_forecast,
    window_train
):
    """
    Retain only true forecast horizons (exclude re-used training points).

    Parameters
    ----------
    T, V : ndarray
        Stacked times and forecast values.
    window_forecast : ndarray
        Full forecast window.
    window_train : ndarray
        Time window of observation.

    Returns
    -------
    T_trunc, V_trunc : ndarray
        Truncated forecast samples.
    """

    block_dim = len(window_forecast)
    last_n = block_dim - len(window_train) + 1

    T_trunc, V_trunc = [], []

    for i in range(0, len(T), block_dim):
        end = i + block_dim
        if end <= len(T):
            T_trunc.extend(T[end - last_n:end])
            V_trunc.extend(V[end - last_n:end])

    return np.array(T_trunc), np.array(V_trunc)

def summarize_forecast(
    df,
    error_width=90
):
    """
    Compute mean forecast and prediction interval.

    Parameters
    ----------
    df : DataFrame
        Forecast samples with columns ['t', 'v'].
    error_width : int
        Width of the prediction interval (e.g. 90).

    Returns
    -------
    summary : DataFrame
        Mean, lower and upper bounds per time step.
    """
    
    alpha = (100 - error_width) / 2

    summary = (
        df.groupby("t")["v"]
        .agg(
            mean="mean",
            lower=lambda x: np.percentile(x, alpha),
            upper=lambda x: np.percentile(x, 100 - alpha),
        )
        .reset_index()
    )

    return summary

def compute_quantiles(
    df,
    quantiles,
    rename_map
):
    """
    Compute quantile-based probabilistic forecasts.

    Parameters
    ----------
    df : DataFrame
        Forecast samples.
    quantiles : list
        Quantile levels to compute.
    rename_map : dict
        Mapping from quantile to column name.

    Returns
    -------
    DataFrame
        Quantile forecasts per time step.
    """
    
    return (
        df.groupby("t")["v"]
        .quantile(quantiles)
        .unstack()
        .rename(columns=rename_map)
        .reset_index()
    )

def run_probabilistic_forecast_season(
    t0_range,
    chains_dir,
    dt_train,
    dt_forecast,
    pinn_model,
    dt,
    times,
    quantiles = quantiles,
    rename_map = rename_map,
    sample_strategy="random"
):
    """
    Run probabilistic forecasting over an entire influenza season.

    Parameters
    ----------
    t0_range : iterable
        Training cutoffs.
    chains_dir : str
        Directory containing MCMC chains.
    dt_train : int
        Time window of observations length.
    dt_forecast : int
        Forecast horizon.
    pinn_model : torch.nn.Module
        Pretrained PINN.
    dt : int
        Time discretization.
    times : array-like
        Continuous time grid.
    sample_strategy : str
        Strategy for posterior sampling.

    Returns
    -------
    forecasts : dict
        Dictionary containing raw samples, summaries and quantiles.
    """ 

    # Container for all forecasts indexed by training cutoff
    forecasts = {}

    # Loop over rolling training windows
    for t0_train in t0_range:
        # Load MCMC posterior samples for current cutoff
        chain_path = f"{chains_dir}/obs={t0_train}.pickle"
        chain = load_mcmc_chain(
            chain_path,
            sample_strategy=sample_strategy
        )

        # Generate posterior predictive trajectories
        T, V, window_forecast = generate_forecast_samples(
            t0_train,
            dt_train,
            dt_forecast,
            chain,
            pinn_model,
            times,
            dt
        )

        # Identify time window of observations (used to truncate forecasts)
        window_train = np.arange(t0_train - dt_train + 1, t0_train + 1)
        window_train = window_train[window_train >= 0]

        # Keep only the true forecast horizon. This removes samples corresponding to re-used data points
        T_tr, V_tr = truncate_forecast_blocks(
            T, V, window_forecast, window_train
        )

        # Build DataFrame of posterior predictive samples
        df = pd.DataFrame({"t": T_tr, "v": V_tr})

        # Summarize uncertainty
        # - raw samples: full posterior predictive distribution
        # - summary_90: mean + 90% prediction interval
        # - summary_50: mean + 50% prediction interval
        # - quantiles: full set of quantile forecasts (for WIS, etc.)
        summary_90 = summarize_forecast(df, error_width=90)
        summary_50 = summarize_forecast(df, error_width=50)
        quant_df = compute_quantiles(df, quantiles, rename_map)

        forecasts[t0_train] = {
            "raw": df,
            "summary_90": summary_90,
            "summary_50": summary_50,
            "quantiles": quant_df,
        }

    return forecasts

def plot_probabilistic_forecasts_season(
    forecasts,
    season,
    dt_train,
    t0_range=range(4, 22, 2),
    t_epi=None,
    epi_weeks=None,
    error_width=90,
    error_width_1=50,
    ylim=(0, 25),
    figsize=(13, 8),
    dpi=300,
):
    """
    Plot probabilistic influenza forecasts with prediction intervals.

    Parameters
    ----------
    forecasts : dict
        Output of run_probabilistic_forecast_season.
        Expected structure: forecasts[t0]["quantiles"] with columns
        ['t', 'q05', 'q25', 'q50', 'q75', 'q95', ...].
    season : str
        Influenza season (e.g. '2023-2024').
    dt_train : int
        Time window of observations length.
    t0_range : iterable
        Training cutoffs to visualize.
    t_epi : array-like
        Continuous time index for the full season.
    epi_weeks : array-like
        Week labels for x-axis.
    error_width : int
        Width of the main prediction interval (e.g. 90).
    error_width_1 : int
        Width of the inner prediction interval (e.g. 50).
    ylim : tuple
        y-axis limits.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.
    """

    # Load observed incidence
    _, incidence = load_influenza_season(season=season)

    # Define plotting styles
    color_90 = [1.0, 0.498, 0.055, 0.2] # orange, wide PI
    color_50 = [0.173, 0.627, 0.173, 0.2] # green, narrow PI
    palette = sns.color_palette()

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Title adapts depending on forecast horizon length
    if len(forecasts[t0_range[0]]["quantiles"]["t"])>10:
        fig.suptitle(
            rf"Italian Seasonal Influenza Long-Term Forecasting - {season}",
            fontsize=20,
            fontweight="bold",
        )
    else:
        fig.suptitle(
            rf"Italian Seasonal Influenza Forecasting - {season}",
            fontsize=20,
            fontweight="bold",
        )

    # Loop over cutoffs (one subplot each)
    for idx, t0_train in enumerate(t0_range):
        # Identify time window of observations
        window_train = np.arange(t0_train - dt_train + 1, t0_train + 1)
        window_train = window_train[window_train >= 0]

        # Extract quantile forecasts
        data_q = forecasts[t0_train]["quantiles"]

        # Plot observed incidence with last observed point highlighted
        ax = fig.add_subplot(3, 3, idx + 1)
        a = ax.plot(t_epi, incidence, ".-", alpha=0.2)
        ax.plot(
            t_epi[window_train],
            incidence[window_train],
            ".-",
            color=a[0].get_color(),
        )
        ax.plot(
            t_epi[t0_train],
            incidence[t0_train],
            "o",
            color=a[0].get_color(),
        )[0].set_label("_nolegend_")

        # Plot prediction intervals
        ax.fill_between(
            data_q["t"],
            data_q["q05"],
            data_q["q95"],
            color=color_90,
            label=f"{error_width}% PI",
        )
        ax.fill_between(
            data_q["t"],
            data_q["q25"],
            data_q["q75"],
            color=color_50,
            label=f"{error_width_1}% PI",
        )

        # Plot posterior predictive mean 
        mean_vals = (
            forecasts[t0_train]["raw"]
            .groupby("t")["v"]
            .mean()
            .reset_index()
        )
        ax.plot(
            mean_vals["t"],
            mean_vals["v"],
            color=palette[2],
        )

        # Median
        #sns.lineplot(
        #    data=data_q,
        #    x="t",
        #    y="q50",
        #    color=palette[2],
        #    legend=False,
        #)

        # Axes formatting
        ax.set_xticks(t_epi)
        ax.set_xticklabels(epi_weeks)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis="both", which="major", labelsize=11)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        
        # Legend only once to avoid clutter
        if idx == 0:
            ax.legend(
                ["All data", "Observations", f"{error_width}% PI", f"{error_width_1}% PI", "Mean"],
                loc="upper right",
            )

        # Label axes only on outer plots
        if idx / 3 >= 1:
            ax.set_xlabel("Week", fontsize=15)

        if idx % 3 == 0:
            ax.set_ylabel("ILI Incidence", fontsize=15)

        ax.set_ylim(*ylim)

    fig.tight_layout()
    
    return fig