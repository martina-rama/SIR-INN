"""
Utility functions for epidemiological analysis and visualization.

This module provides tools to:
- compute incidence from SIR trajectories,
- qualitatively assess SIR-INN reconstructions,
- visualize learned epidemic dynamics,
- select representative epidemic scenarios.

The functions are primarily used for diagnostic
and qualitative evaluation of trained SIR-INN models.
"""

# ------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------

def incidence_from_sir(S, times, dt=1, scale=1000):
    """
    Compute incidence from S trajectories.

    Parameters
    ----------
    S : array-like
        Susceptible trajectories (absolute numbers).
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
    """

    # Styling
    SIR_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    SIR_labels = ['S', 'I', 'R']
    color_true_inc = "#D55E00"
    color_pred_inc = "#0072B2"

    # Grid of subplots (up to 12 scenarios)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    title_map = {
        'SIR': 'SIR dynamics',
        'I': 'Infectious dynamics',
        'incidence': 'ILI incidence infectious dynamics'
    }
    fig.suptitle(title_map.get(plot_type, "SIR dynamics"), fontsize=16)

    # Loop over selected epidemic scenarios
    for i, idx in enumerate(idx_plot):
        ax = axes[i // 3, i % 3]
        
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
                ax.legend([
                    plt.Line2D([0], [0], color=SIR_colors[1], lw=2, label="I true"),
                    plt.Line2D([0], [0], color=SIR_colors[1], lw=2, ls='--', label="I pred")
                ], fontsize=9)

            else:
                ax.legend(['SIR incidence', 'PINN incidence'], fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

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
