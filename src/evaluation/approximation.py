"""
Utility functions for epidemiological analysis and visualization, and to validate SIR approximation ability of the pre-trained SIR-INN.
"""

# ------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import torch
import functorch # required for higher-order autodiff in PINNs
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ------------------------------------------------------------------

def incidence_from_sir(S, times, dt=1, scale=1000):
    """
    Compute incidence from S trajectories.

    Parameters
    ----------
    S : array-like
        Normalized susceptible trajectories.
    times : array-like
        Time points at which to compute incidence.
    dt : int, optional
        Time step to compute incidence over (default 1, i.e., daily, as the training set).
    scale : float, optional
        Scaling factor for incidence (default 1000) to match ILI reporting units.

    Returns
    -------
    incidence : np.ndarray
        Incidence at specified time points, scaled.
    """
    
    # Discrete derivative of the susceptible compartment
    S_diff = np.diff(S) * scale  
    
    incidence = []

    # For each requested time point, aggregate new infections over the previous dt steps
    for t in times:
        incidence.append(-np.sum(S_diff[t-dt:t])) 
       
    incidence = np.array(incidence)    
    
    return incidence

def evaluation_pinn(x,model,times):
    """
    Generate a PINN evaluation at the given time points for a single parameter set.

    Parameters
    ----------
    x : array-like
        Epidemiological parameters [beta, gamma, t0].
    model : torch.nn.Module
        Pretrained SIR-INN model.
    times : array-like
        Continuous time grid used for PINN evaluation.

    Returns
    -------
    model_eval : ndarray
        Forward evaluation of the PINN in all the time domain.
    """

    # Build PINN input: [time, beta, gamma]
    input_eval = torch.tensor(np.hstack([times[:, None], np.tile([x[0], x[1]], (len(times), 1))])).float()

    model.eval()
    # Forward evaluation of the PINN
    with torch.no_grad():
        model_eval = model(input_eval).numpy()

    return model_eval

def sir_ode(y, t, beta, gamma, N=1e6):
    """
    Standard SIR differential equations.
    
    Parameters
    ----------
    y : array [S, I, R]
        Current state.
    t : float
        Current time.
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    N : float
        Total population.
    
    Returns
    -------
    dydt : array [dS/dt, dI/dt, dR/dt]
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def solve_sir_ode(t_array, beta, gamma, S0, I0, R0, N=1e6):
    """
    Solve SIR using numerical integration.
    
    Parameters
    ----------
    t_array : array
        Time points for evaluation.
    beta, gamma : float
        SIR parameters.
    S0, I0, R0 : float
        Initial conditions.
    N : float
        Total population.
    
    Returns
    -------
    solution : array of shape (len(t_array), 3)
        SIR trajectories [S, I, R].
    """
    y0 = [S0, I0, R0]
    solution = odeint(sir_ode, y0, t_array, args=(beta, gamma, N))
    return solution

def plot_sir_inn_learning(
    x_train_np,
    idx_train,
    idx_plot,
    S_obs, I_obs, R_obs,
    S_pred, I_pred, R_pred,
    plot_type='SIR',
    N=1e6,
    dt=1,
    scale=1000
):
    """
    Visualize qualitative SIR-INN reconstructions.

    Depending on plot_type, the function compares:
    - full SIR dynamics,
    - infectious compartment only,
    - ILI incidence reconstructed from S.

    The comparison is qualitative and intended to assess
    epidemiological plausibility.
    
    Parameters
    ----------
    x_train_np : np.ndarray
        Training input array of shape (N, 3) with columns [time, beta, gamma].
    idx_train : np.ndarray
        Scenario indices associated with each training point.
    idx_plot : array-like
        Subset of scenario indices to visualize (up to 12).
    S_obs, I_obs, R_obs : np.ndarray
        Observed SIR compartments from the training set.
    S_pred, I_pred, R_pred : np.ndarray
        SIR compartments predicted by the SIR-INN.
    plot_type : str, optional
        What to visualize: 'SIR' (full dynamics), 'I' (infectious only),
        or 'incidence' (reconstructed ILI incidence). Default: 'SIR'.
    N : float, optional
        Total population, used for incidence scaling. Default: 1e6.
    dt : int, optional
        Time step for incidence aggregation. Default: 1.
    scale : float, optional
        Scaling factor applied to incidence values. Default: 1000.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the (4 x 3) panel of trajectory comparisons.
    """
    
    # Styling
    SIR_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    SIR_labels = ['S', 'I', 'R']
    color_true_inc = "#D55E00"
    color_pred_inc = "#0072B2"

    n_plots = len(idx_plot)
    n_cols = 3
    n_rows = 4

    # Grid of subplots (up to 12 scenarios)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    title_map = {
        'SIR': 'SIR dynamics',
        'I': 'Infectious dynamics',
        'incidence': 'ILI incidence infectious dynamics'
    }
    fig.suptitle(title_map.get(plot_type, "SIR dynamics"), fontsize=16)

    # Loop over selected epidemic scenarios
    for i, idx in enumerate(idx_plot):
        row = i // n_cols  
        col = i % n_cols   
        ax = axes[row, col] 

        # Indices corresponding to the selected scenario
        idx_p = np.where(idx_train == idx)[0]

        # Extract epidemiological parameters
        beta = x_train_np[idx_p[0], 1]
        gamma = x_train_np[idx_p[0], 2]
        R0 = beta / gamma

        # Plot full SIR dynamics
        if plot_type == 'SIR':
            ax.plot(x_train_np[idx_p, 0], S_obs[idx_p], color=SIR_colors[0], lw=2)
            ax.plot(x_train_np[idx_p, 0], S_pred[idx_p], '--', color=SIR_colors[0], lw=2)

            ax.plot(x_train_np[idx_p, 0], I_obs[idx_p], color=SIR_colors[1], lw=2)
            ax.plot(x_train_np[idx_p, 0], I_pred[idx_p], '--', color=SIR_colors[1], lw=2)

            ax.plot(x_train_np[idx_p, 0], R_obs[idx_p], color=SIR_colors[2], lw=2)
            ax.plot(x_train_np[idx_p, 0], R_pred[idx_p], '--', color=SIR_colors[2], lw=2)

            ax.set_title(r'$R_0$=%.2f' % R0)

        # Plot only infectious compartment
        elif plot_type == 'I':
            ax.plot(x_train_np[idx_p, 0], I_obs[idx_p], color=SIR_colors[1], lw=2)
            ax.plot(x_train_np[idx_p, 0], I_pred[idx_p], '--', color=SIR_colors[1], lw=2)
            ax.set_title(r'$R_0$=%.2f' % R0)

        # Plot reconstructed ILI incidence
        elif plot_type == 'incidence':
            times = np.arange(len(idx_p))

            # Incidence from observed and predicted S
            inc_true = incidence_from_sir(S_obs[idx_p], times)
            inc_pred = incidence_from_sir(S_pred[idx_p], times)

            ax.plot(x_train_np[idx_p, 0], inc_true, color=color_true_inc, lw=2.5)
            ax.plot(x_train_np[idx_p, 0], inc_pred, '--', color=color_pred_inc, lw=2.5)
            ax.set_title(r'$R_0$=%.2f' % R0)

        ax.grid(True)

        # Legend (shown only once)
        if i == 0:
            if plot_type == 'SIR':
                handles = []
                for j, c in enumerate(SIR_colors):
                    handles.append(plt.Line2D([0], [0], color=c, lw=2, label=f"{SIR_labels[j]} true"))
                    handles.append(plt.Line2D([0], [0], color=c, lw=2, ls='--', label=f"{SIR_labels[j]} pred"))
                ax.legend(handles=handles, fontsize=9)

            elif plot_type == 'I':
                handles = [
                    plt.Line2D([0], [0], color=SIR_colors[1], lw=2, label="I true"),
                    plt.Line2D([0], [0], color=SIR_colors[1], lw=2, ls='--', label="I pred")
                ]
                ax.legend(handles=handles, fontsize=9)

            else:
                handles = [
                    plt.Line2D([0], [0], color=color_true_inc, lw=2.5, label="SIR incidence"),
                    plt.Line2D([0], [0], color=color_pred_inc, lw=2.5, ls='--', label="PINN incidence")
                ]
                ax.legend(handles=handles, fontsize=9)  

        # X-axis label only on last row
        
        if row == n_rows - 1:
            ax.set_xlabel("Time (days)", fontsize=10)
        else:
            ax.tick_params(labelbottom=False)

        # Y-axis label only on first column
        if col == 0:
            ax.set_ylabel("Population", fontsize=10)
        else:
            ax.tick_params(labelleft=False)


    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    return fig

def plot_sir_conservation(
    x_train_np,
    idx_train,
    idx_plot,
    S_pred, I_pred, R_pred,
):

    """
    Plot population conservation (S + I + R ≈ 1) for a subset of training scenarios.
    
    For each selected scenario, shows the sum S+I+R predicted by the SIR-INN
    over time, together with a reference line at 1. The subplot title reports
    the MAE and maximum absolute error for that scenario.
    
    Parameters
    ----------
    x_train_np : np.ndarray
        Training input array of shape (N, 3) with columns [time, beta, gamma].
    idx_train : np.ndarray
        Scenario indices associated with each training point.
    idx_plot : array-like
        Subset of scenario indices to visualize (up to 12).
    S_pred, I_pred, R_pred : np.ndarray
        SIR compartments predicted by the SIR-INN.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the (4 x 3) conservation panel.
    """
    
    color_sum = "#8B008B"
    color_ref = "#888888"

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle("Population Conservation: S + I + R ≈ N", fontsize=16)
    
    n_plots = len(idx_plot)
    n_cols = 3
    n_rows = 4


    for i, idx in enumerate(idx_plot):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        idx_p = np.where(idx_train == idx)[0]
        t = x_train_np[idx_p, 0]

        beta  = x_train_np[idx_p[0], 1]
        gamma = x_train_np[idx_p[0], 2]
        R0    = beta / gamma

        N_sum = S_pred[idx_p] + I_pred[idx_p] + R_pred[idx_p]
        mae   = np.mean(np.abs(N_sum - 1.0))
        max_e = np.max(np.abs(N_sum - 1.0))

        # Reference line
        ax.axhline(1.0, color=color_ref, lw=1.5, ls='--', label='N = 1')
        # S+I+R curve
        ax.plot(t, N_sum, color=color_sum, lw=2, label='S+I+R')

        ax.set_title(
            r'$\mathbf{R_0 = %.2f}$' % R0 + '\n' +
            r'MAE=%.2e   max|err|=%.2e' % (mae, max_e),
            fontsize=10,
            loc='center'
        )

        ax.set_ylim([max(0, N_sum.min() - 0.05), N_sum.max() + 0.05])
        ax.grid(True)

        # X-axis label only on last row
        if row == n_rows - 1:
            ax.set_xlabel("Time (days)", fontsize=10)
        else:
            ax.tick_params(labelbottom=False)

        # Y-axis label only on first column
        if col == 0:
            ax.set_ylabel("S + I + R", fontsize=10)
        else:
            ax.tick_params(labelleft=False)
        
        if i == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return fig

def compute_conservation_error(
    idx_train,
    S_pred, I_pred, R_pred,
):
    """
    Compute population conservation error (S+I+R ≈ 1) 
    across the entire training set and per scenario.

    Parameters
    ----------
    idx_train : np.ndarray
        Scenario indices associated with each training point.
    S_pred, I_pred, R_pred : np.ndarray
        SIR compartments predicted by the SIR-INN across all training points.
    
    Returns
    -------
    df : pd.DataFrame
        Per-scenario conservation error with columns MAE, max_err, std_err,
        indexed by scenario index.
    """
    
    N_sum = S_pred + I_pred + R_pred
    abs_err = np.abs(N_sum - 1.0)

    scenarios = np.unique(idx_train)
    results = []

    for idx in scenarios:
        idx_p = np.where(idx_train == idx)[0]
        err = abs_err[idx_p] # for all the time domain
        results.append({
            'scenario': idx,
            'MAE':      np.mean(err),
            'max_err':  np.max(err),
            'std_err':  np.std(err),
        })

    df = pd.DataFrame(results).set_index('scenario')

    # Global stats across all training points
    print("=" * 45)
    print("  Conservation Error — Full Training Set")
    print("=" * 45)
    print(f"  Global MAE     : {abs_err.mean():.4e}")
    print(f"  Global max|err|: {abs_err.max():.4e}")
    print(f"  Global std     : {abs_err.std():.4e}")
    print(f"  # scenarios    : {len(scenarios)}")
    print(f"  # total points : {len(N_sum)}")
    print("=" * 45)

    return df

def sample_scenarios_by_R0(x_train_np, idx_train, R0_min=1.1, n_samples=12):
    """
    Randomly sample epidemic scenarios with R0 above a threshold.

    This is used to visualize meaningful epidemic trajectories,
    excluding near-subcritical or trivial dynamics.

    Parameters
    ----------
    x_train_np : np.ndarray
        Training input array containing time, beta and gamma.
    idx_train : np.ndarray
        Scenario indices associated with training points.
    R0_min : float
        Minimum reproduction number threshold.
    n_samples : int
        Number of scenarios to sample.

    Returns
    -------
    idx_plot : np.ndarray
        Selected scenario indices.
    """
    
    valid_idx = []

    # Loop over unique epidemic scenarios
    for idx in np.unique(idx_train): 

        # Retrieve parameters for the scenario
        i0 = np.where(idx_train == idx)[0][0]
        beta = x_train_np[i0, 1]
        gamma = x_train_np[i0, 2]

        # Keep only sufficiently supercritical epidemics
        if beta / gamma > R0_min:
            valid_idx.append(idx)

    # Sample without replacement
    n_samples = min(n_samples, len(valid_idx))
    
    return np.random.choice(valid_idx, size=n_samples, replace=False)

def sirinn_error_grid(
    sir_inn,
    beta_min, beta_max,
    gamma_min, gamma_max,
    idx_train,
    x_train_np,
    times,
    N=1e6,
    n_grid=10
):
    
    """
    Compute SIR-INN mean squared error over a regular (beta, gamma) grid.
    
    For each grid point, solves the SIR ODE numerically and compares it
    to the SIR-INN prediction. The error is averaged across S, I, and R
    compartments and across all time steps.
    
    Parameters
    ----------
    sir_inn : torch.nn.Module
        Pretrained SIR-INN model.
    beta_min, beta_max : float
        Range of transmission rates for the grid.
    gamma_min, gamma_max : float
        Range of recovery rates for the grid.
    idx_train : np.ndarray
        Scenario indices, used to extract training (beta, gamma) pairs.
    x_train_np : np.ndarray
        Training input array of shape (N, 3) with columns [time, beta, gamma].
    times : array-like
        Time grid used for ODE integration and PINN evaluation.
    N : float, optional
        Total population. Default: 1e6.
    n_grid : int, optional
        Number of grid points per axis. Default: 10.
    
    Returns
    -------
    error_grid : np.ndarray
        (n_grid x n_grid) array of mean squared errors.
    betas_train : np.ndarray
        Beta values from the training set.
    beta_vals : np.ndarray
        Beta grid values.
    gammas_train : np.ndarray
        Gamma values from the training set.
    gamma_vals : np.ndarray
        Gamma grid values.
    """
    
    betas_train = []
    gammas_train = []

    for idx in np.unique(idx_train):
        i0 = np.where(idx_train == idx)[0][0]
        betas_train.append(x_train_np[i0, 1])
        gammas_train.append(x_train_np[i0, 2])

    betas_train = np.array(betas_train)
    gammas_train = np.array(gammas_train)

    # parameter grid
    beta_vals = np.linspace(beta_min, beta_max, n_grid)
    gamma_vals = np.linspace(gamma_min, gamma_max, n_grid)

    error_grid = np.zeros((n_grid, n_grid))

    for i, beta in enumerate(beta_vals):
        for j, gamma in enumerate(gamma_vals):

            # True SIR
            I0 = 1.0
            S0 = N - I0
            R0_init = 0.0

            sol = solve_sir_ode(times, beta, gamma, S0, I0, R0_init, N)

            S_true = sol[:,0] / N
            I_true = sol[:,1] / N
            R_true = sol[:,2] / N

            # PINN prediction
            params_epi = np.array([beta, gamma])
            model_eval = evaluation_pinn(params_epi, sir_inn, times)

            S_pred = model_eval[:,0]
            I_pred = model_eval[:,1]
            R_pred = model_eval[:,2]

            # Errors
            err_S = np.mean((S_pred - S_true)**2)
            err_I = np.mean((I_pred - I_true)**2)
            err_R = np.mean((R_pred - R_true)**2)

            error_grid[j, i] = (err_S + err_I + err_R) / 3

    return error_grid, betas_train, beta_vals, gammas_train, gamma_vals

def sirinn_error_heatmap(
    error_grid, 
    betas_train, 
    beta_vals, 
    gammas_train, 
    gamma_vals,
    vmin=1e-8,
    vmax=0.05
):
    """
    Plot a heatmap of SIR-INN approximation error over the (beta, gamma) grid.
    
    Cells corresponding to training set parameter pairs are highlighted
    with a red border. The color scale is logarithmic.
    
    Parameters
    ----------
    error_grid : np.ndarray
        (n_grid x n_grid) array of errors, as returned by sirinn_error_grid.
    betas_train : np.ndarray
        Beta values from the training set (used to mark training cells).
    beta_vals : np.ndarray
        Beta grid values for axis labels.
    gammas_train : np.ndarray
        Gamma values from the training set.
    gamma_vals : np.ndarray
        Gamma grid values for axis labels.
    vmin, vmax : float, optional
        Color scale bounds. Default: 1e-8, 0.05.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Heatmap figure.
    """
    
    norm = LogNorm(vmin=vmin, vmax=vmax)
        
    fig, ax = plt.subplots(figsize=(15,15))
    ax = sns.heatmap(
        error_grid,
        cmap="YlGnBu",
        norm=norm,
        xticklabels=[round(float(x),2) for x in beta_vals],
        yticklabels=[round(float(x),2) for x in gamma_vals],
        annot=True,
        fmt=".2e",                      
        annot_kws={"size":12},           
        cbar_kws={"label":"Mean squared error","shrink": 0.75,"pad": 0.03,"aspect": 30},
        square=True
        )
    
    cbar = ax.collections[0].colorbar
    cbar.set_label("Mean squared error", labelpad=15, fontsize=15)

    # Write the parameters in pairs
    train_pairs = [(round(b,2), round(g,2)) for b,g in zip(betas_train, gammas_train)]
    
    for i, gamma in enumerate(gamma_vals):
        for j, beta in enumerate(beta_vals):
    
            beta_r = np.round(beta,2)
            gamma_r = np.round(gamma,2)
    
            if (beta_r, gamma_r) in train_pairs:
    
                rect = patches.Rectangle(
                    (j, i),      
                    1,           
                    1,          
                    fill=False,
                    edgecolor="red",
                    linewidth=1.8
                )
    
                ax.add_patch(rect)
    
    ax.invert_yaxis()
    ax.set_xlabel(r"$\beta$",fontsize = 18, fontweight ='bold',labelpad = 9)
    ax.set_ylabel(r"$\gamma$",fontsize = 18, fontweight ='bold',labelpad = 9)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    ax.set_title("SIR-INN approximation error over parameter space",fontsize = 20, pad = '20')
    
    legend_elements = [
        Line2D([0],[0], color='red', lw=2, label='Training set')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right',bbox_to_anchor=(1.17,-0.008),fontsize=10)

    return fig