"""
Probabilistic forecasting utilities based on MCMC posterior samples and the pretrained SIR-INN model.
"""

# ------------------------------------------------------------------

# Make repository importable when running as script or from notebook
import sys
from pathlib import Path
# Add repository root to Python path
repo_root = Path().resolve().parent
sys.path.append(str(repo_root))

from src.evaluation.approximation import incidence_from_sir, evaluation_pinn
from src.forecasting.inference import cut_times, load_mcmc_chain, load_runtimes
from src.data.data_loader import load_influenza_season
from src.utils.constants import t0_range,t_epi,epi_weeks,times,quantiles,rename_map

import torch
import functorch # required for higher-order autodiff in PINNs
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch

# ------------------------------------------------------------------

def forecast_pinn(x, model, times, dt, window):
    """
    Generate an incidence forecast from the PINN for a single parameter set.

    Parameters
    ----------
    x : array-like
        Epidemiological parameters [beta, gamma, t0].
    model: torch.nn.Module
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

    # Forward evaluation of the PINN in all the time domain
    model_eval = evaluation_pinn(x,model,times)

    # Discrete times at which incidence is computed
    t_inc = times[0:590:dt].astype('i')

    # Cut time axis based on epidemic onset t0
    t_inc,y_inc = cut_times(t_inc, window, x[2], forecast = True)

    # Compute incidence from susceptible trajectory
    incidence_estimated = incidence_from_sir(S=model_eval[:, 0], times=t_inc, dt=dt)
    
    return incidence_estimated


def forecast_TV(time_plot_x,chain_samples,times,dt,window_forecast,pinn_model=None):
    """
    Generate stacked forecast trajectories over multiple posterior samples.

    Parameters
    ----------
    time_plot_x : array-like
        Time indices used for plotting.
    chain_samples : ndarray
        Sampled parameter vectors from the posterior.
    times : array-like
        Continuous time grid.
    dt : int
        Time discretization.
    window_forecast : array-like
        Forecast window indices.
    forecast_model : str
        'pinn' or 'ode' - which model to use for forecasting.
    pinn_model : torch.nn.Module, optional
        Pretrained SIR-INN model (required if forecast_model='pinn').

    Returns
    -------
    T : ndarray
        Stacked time indices.
    V : ndarray
        Stacked incidence values.
    """
    
    forecast = {'t': [], 'v': []}
    
    # Validate inputs
    if pinn_model is None:
        raise ValueError("pinn_model must be provided when forecast_model='pinn'")
    
    forecast_fn = lambda params: forecast_pinn(
        params, pinn_model, times, dt, len(window_forecast)
    )

    # Initialize with first sample, then stack remaining trajectories
    forecast['t'] = time_plot_x
    forecast['v'] = forecast_fn(chain_samples[0])
    
    # Loop over remaining posterior samples
    for params in chain_samples[1:]:
        forecast['t'] = np.concatenate((forecast['t'], time_plot_x), axis=0)
        forecast['v'] = np.concatenate((forecast['v'], forecast_fn(params)), axis=0)
    
    # Flatten for DataFrame construction
    T = np.vstack(forecast['t'])[:, 0]
    V = np.vstack(forecast['v'])[:, 0]
    
    return T,V

def generate_forecast_samples(
    t0_train,
    dt_train,
    dt_forecast,
    chain_samples,
    times,
    dt,
    pinn_model=None,
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
    times : array-like
        Continuous time grid.
    dt : int
        Time discretization.
    pinn_model : torch.nn.Module, optional
        Pretrained PINN (required if forecast_model='pinn').
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
        times,
        dt,
        window_forecast,
        pinn_model=pinn_model,
    )

    return T, V, window_forecast

def truncate_forecast_blocks(
    T,
    V,
    window_forecast,
    window_train
):
    """
    Retain only true forecast horizons.

    Parameters
    ----------
    T, V : ndarray
        Stacked times and forecast values.
    window_forecast : ndarray
        Full forecast window.
    window_train : ndarray
        Array of observation window indices, used to determine how many samples to discard from each forecast block

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
    chains_dir,
    dt_train,
    dt_forecast,
    dt=7,
    times=times,
    t0_range=t0_range,
    priors='uniform', 
    pinn_model=None,
    fit=False,
    quantiles = quantiles,
    rename_map = rename_map,
    sample_strategy="tail",
    nsample=1000,
    nsim=10000

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
    times : array-like
        Continuous time grid.
    dt : int
        Time discretization.
    pinn_model : torch.nn.Module, optional
        Pretrained PINN (required if forecast_model='pinn').
    quantiles : list
        Quantile levels to compute.
    rename_map : dict
        Mapping from quantile to column name.
    sample_strategy : str
        Strategy for posterior sampling.
    nsample : int
        Number of samples for random strategy.
    nsim : int
        Total MCMC iterations.
    priors : str, optional
        Prior type used during inference ('uniform' or 'normal'). Default: 'uniform'.
    fit : bool, optional
        If True, includes the observation window in the forecast output
        (posterior predictive interpolation). If False, only the true
        forecast horizon is retained. Default: False.

    Returns
    -------
    forecasts : dict
        Dictionary containing raw samples, summaries and quantiles.
    """ 

    # Create output directory
    output_subdir = f"{chains_dir}/{priors}/forecast"
    os.makedirs(output_subdir, exist_ok=True)
    
    # Validate inputs
    if pinn_model is None:
        raise ValueError("pinn_model must be provided when forecast_model='pinn'")
    
    # Container for all forecasts indexed by training cutoff
    forecasts = {}

    # Loop over rolling training windows
    for t0_train in t0_range:
        # Load MCMC posterior samples for current cutoff
        # Chains are stored in subdirectories by inference model type
        chain_path = f"{chains_dir}/{priors}/obs={t0_train}.pickle"
        chain = load_mcmc_chain(
            chain_path,
            sample_strategy=sample_strategy,
            nsample=nsample,
            nsim=nsim
        )

        # Generate posterior predictive trajectories
        T, V, window_forecast = generate_forecast_samples(
            t0_train,
            dt_train,
            dt_forecast,
            chain,
            times,
            dt,
            pinn_model=pinn_model,
        )

        # Identify time window of observations (used to truncate forecasts)
        window_train = np.arange(t0_train - dt_train + 1, t0_train + 1)
        window_train = window_train[window_train >= 0]

        if fit==True:
            # Fitting and forecasting statistics
            T_tr,V_tr = T,V

        else:
            # Keep only the forecast horizon for the statistics. This removes samples corresponding to the observed data points
            T_tr, V_tr = truncate_forecast_blocks(
                T, V, window_forecast, window_train
            )

        
        # Build DataFrame of posterior predictive samples
        df = pd.DataFrame({"t": T_tr, "v": V_tr})

        summary_90 = summarize_forecast(df, error_width=90)
        summary_50 = summarize_forecast(df, error_width=50)
        quant_df = compute_quantiles(df, quantiles, rename_map)

        forecasts[t0_train] = {
            "raw": df,
            "summary_90": summary_90,
            "summary_50": summary_50,
            "quantiles": quant_df,
        }

        # Save the forecasts
        with open(f"{output_subdir}/forecast_results.pickle", "wb") as f:
            pickle.dump(forecasts, f)

    return forecasts

def plot_probabilistic_forecasts_season(
    forecasts,
    season,
    dt_train,
    grid_size=3,
    t_epi=t_epi,
    epi_weeks=epi_weeks,
    error_width=90,
    error_width_1=50,
    ylim=(0, 25),
    figsize=(13, 8),
    dpi=300
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
    grid_size : int
        Controls the subplot grid layout and which cutoffs are visualized.
        Accepted values: 2, 3, or 5. A (grid_size x grid_size) panel is produced,
        with cutoffs selected automatically based on the season.
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

    if grid_size==3:
        t0_range=range(4, 22, 2)
    elif grid_size==5:
        t0_range=range(3, 28)
    elif grid_size==2 and season=='2024-2025':
        t0_range=range(12,16)
        #t0_range=range(10,14)
    elif grid_size==2 and season=='2023-2024':
        t0_range=range(8,12)
        #t0_range=range(6,10)
    else:
       raise ValueError("Number of subplots not valid. Please select either 2,3 or 5 as the grid size of subplots to visualize.")

    font_scale = {2: 1.0, 3: 0.95, 5: 0.7}
    marker_scale = {2: 1.6,  3: 1.0,  5: 0.6}
    fs = font_scale.get(grid_size, 0.85)
    ms = marker_scale.get(grid_size, 1.0)
    
    label_fontsize   = int(15 * fs)
    tick_fontsize    = int(11 * fs)
    legend_fontsize  = int(10 * fs)
    suptitle_fontsize = int(20 * fs)
    major_locator    = {2: 3, 3: 5, 5: 4}.get(grid_size, 5)

    markersize_main   = 6  * ms   
    markersize_obs    = 6  * ms   
    markersize_last   = 6  * ms   
    linewidth_data    = 1.2 * ms

    # Load observed incidence
    _, incidence = load_influenza_season(season=season)

    # Define plotting styles
    color_90 = [1.0, 0.498, 0.055, 0.2] # orange, wide PI
    color_50 = [0.173, 0.627, 0.173, 0.2] # green, narrow PI
    palette = sns.color_palette()

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Title adapts depending on forecast horizon length
    title_txt = (
        rf"Italian Seasonal Influenza Long-Term Forecasting - {season}"
        if len(forecasts[t0_range[0]]["quantiles"]["t"]) > 10
        else rf"Italian Seasonal Influenza Forecasting - {season}"
    )
    fig.suptitle(title_txt, fontsize=suptitle_fontsize, fontweight="bold")
    
    n_plots = len(list(t0_range))

    # Loop over cutoffs (one subplot each)
    for idx, t0_train in enumerate(t0_range):
        # Identify time window of observations
        window_train = np.arange(t0_train - dt_train + 1, t0_train + 1)
        window_train = window_train[window_train >= 0]

        # Extract quantile forecasts
        data_q = forecasts[t0_train]["quantiles"]

        # Plot observed incidence with last observed point highlighted
        ax = fig.add_subplot(grid_size, grid_size, idx + 1)
        
        a = ax.plot(
            t_epi, incidence, ".-",
            alpha=0.2,
            markersize=markersize_main,
            linewidth=linewidth_data
        )

        ax.plot(
            t_epi[window_train], incidence[window_train], ".-",
            color=a[0].get_color(),
            markersize=markersize_obs,
            linewidth=linewidth_data
        )

        ax.plot(
            t_epi[t0_train], incidence[t0_train], "o",
            color=a[0].get_color(),
            markersize=markersize_last
        )
 
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
            linewidth=linewidth_data
        )

        # Alternative: use median instead of mean
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
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        #ax.tick_params(axis="both", which="major", labelsize=11)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        
        # Legend only once to avoid clutter
        if idx == 0:
            legend_elements = [
                plt.Line2D([0], [0], color=a[0].get_color(), linestyle='-', marker='.', 
                           markersize=legend_fontsize*0.8, alpha=0.2, label='All data'),
                plt.Line2D([0], [0], color=a[0].get_color(), linestyle='-', marker='.', 
                           markersize=legend_fontsize*0.8, label='Observations'),
                Patch(facecolor=color_90, edgecolor='none', label=f'{error_width}% PI'),
                Patch(facecolor=color_50, edgecolor='none', label=f'{error_width_1}% PI'),
                plt.Line2D([0], [0], color=palette[2], linewidth=linewidth_data*1.3, label='Mean'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=legend_fontsize)

        if idx + 1 > n_plots - grid_size:
            ax.set_xlabel("Week", fontsize=label_fontsize, labelpad=8)

        if idx % grid_size == 0:
            ax.set_ylabel("ILI Incidence", fontsize=label_fontsize, labelpad=8)

        ax.set_ylim(*ylim)

    fig.tight_layout()
    
    return fig