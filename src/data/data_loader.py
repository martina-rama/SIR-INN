"""
Data loading and visualization utilities for italian influenza surveillance.

Data are retrieved directly from official Influnet and Influcast
repositories (without requiring local cloning) to ensure reproducibility and up-to-date access.
"""

# ------------------------------------------------------------------

import torch
import pandas as pd
import io
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import requests

# ---------------------------------------------------------------------

# URLs
INFLUNET_DATA_URL = (
    "https://raw.githubusercontent.com/fbranda/influnet/main/"
    "data-aggregated/epidemiological_data/national_cases.csv"
)
INFLUCAST_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Predizioni-Epidemiologiche-Italia/Influcast/main/"
    "sorveglianza/ILI"
)

# ------------------------------------------------------------------

def load_train_data(path):
    """
    Load SIR-INN training dataset from CSV file.

    The dataset is expected to contain:
    - input features (time, parameters),
    - observed SIR compartments,
    - a scenario index identifying each epidemic realization.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    x : torch.Tensor
        Model inputs (time, beta, gamma).
    y : torch.Tensor
        Observed SIR components (S, I, R).
    idx_train : np.ndarray
        Scenario indices for grouping trajectories.
    """
    
    data = pd.read_csv(path, header=None).values
    x = torch.tensor(data[:, :3]).float()
    y = torch.tensor(data[:, 3:6]).float()
    idx_train = data[:, -1]
    
    return x, y, idx_train

def _convert_year_week_to_epiweeks(year_week_series: pd.Series) -> pd.Series:
    """
    Convert year-week strings into continuous epidemiological weeks.

    Weeks < 42 are shifted by +52 to ensure continuity
    within an influenza season.

    Parameters
    ----------
    year_week_series : pd.Series
        Strings formatted as 'YYYY-WW'.

    Returns
    -------
    epi_week : pd.Series
        Continuous epidemiological week index.
    """
    
    week = year_week_series.apply(lambda x: int(x.split("-")[1]))
    week = week.astype(int)

    # Shift weeks belonging to the following calendar year
    week[week < 42] += 52

    return week

def load_influenza_season(
    season: str,
    data_url: str = INFLUNET_DATA_URL
):
    """
    Load a single influenza season from Influnet aggregated data.

    Parameters
    ----------
    season : str
        Influenza season label (e.g. '2023-2024').
    data_url : str, optional
        URL to the Influnet aggregated CSV file.

    Returns
    -------
    week : np.ndarray
        Continuous epidemiological week index.
    incidence : np.ndarray
        ILI incidence for the selected season.
    """
    data = pd.read_csv(data_url)

    # Create continuous epidemiological week index
    data["week"] = _convert_year_week_to_epiweeks(data["year_week"])

    # Select season
    season_data = data[data["flu_season"] == season]

    week = season_data["week"].values
    incidence = season_data["incidence"].values

    return week, incidence


def load_multiple_influenza_seasons(
    seasons,
    data_url: str = INFLUNET_DATA_URL
):
    """
    Load multiple influenza seasons.

    Parameters
    ----------
    seasons : list[str]
        List of influenza season labels (e.g. ['2022-2023', '2023-2024']).
    data_url : str, optional
        URL to the Influnet aggregated CSV file.

    Returns
    -------
    seasons_data : dict
        Dictionary mapping season labels to dicts with keys:
            - 'week'
            - 'incidence'
    """
    data = pd.read_csv(data_url)
    data["week"] = _convert_year_week_to_epiweeks(data["year_week"])

    seasons_data = {}

    for season in seasons:
        season_data = data[data["flu_season"] == season]

        seasons_data[season] = {
            "week": season_data["week"].values,
            "incidence": season_data["incidence"].values
        }

    return seasons_data

def plot_observed_ili_seasons(
    seasons_data,
    country: str = "Italy",
    top_padding: float = 2.5,
    figsize=(10, 5),
):
    """
    Plot observed ILI incidence for multiple influenza seasons.
    This function plots weekly influenza-like illness incidence
    and highlights the epidemic peak for each season.

    Parameters
    ----------
    seasons_data : dict
        Dictionary mapping season labels to dicts with keys:
            - 'week'       : np.ndarray of epidemiological week indices
            - 'incidence'  : np.ndarray of weekly ILI incidence values
        As returned by load_multiple_influenza_seasons().    
    country : str, optional
        Country name to be displayed in the title.
    top_padding : float, optional
        Vertical padding above the maximum incidence for annotations.
    figsize : tuple, optional
        Figure size.
    """
    
    # Epidemiological week labels (42–52, 01–17)
    week_labels = np.array(list(range(42, 53)) + list(range(1, 18)))
    week_labels_str = np.array([f"{w:02d}" for w in week_labels])

    # Compute peaks and y-limits
    all_incidence = []
    peaks = {}

    for season in seasons_data.keys():
        inc = seasons_data[season]["incidence"]
        all_incidence.append(inc)
        peaks[season] = np.argmax(inc)

    all_incidence = np.concatenate(all_incidence)
    ymin = all_incidence.min() - 0.5
    ymax = all_incidence.max() + top_padding

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", "-.", ":"]

    for i, season in enumerate(seasons_data.keys()):
        inc = seasons_data[season]["incidence"]
        peak_idx = peaks[season]

        ax.plot(
            week_labels_str,
            inc,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            label=f"{season} season",
        )

        # Annotate peak
        ax.annotate(
            f"Peak: {inc[peak_idx]:.1f}",
            xy=(week_labels_str[peak_idx], inc[peak_idx]),
            xytext=(week_labels_str[peak_idx], inc[peak_idx] + 0.7 * top_padding),
            arrowprops=dict(arrowstyle="->"),
            ha="center",
            fontsize=10,
        )

    # Styling
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_title(
        f"Influenza-like illness observations – {country}",
        fontsize=14,
    )
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("ILI incidence rate", fontsize=12)

    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def load_influcast_week(
    season: str,
    country: str,
    week: str
):
    """
    Load ILI incidence for a specific epidemiological week
    directly from the Influcast repository.

    Parameters
    ----------
    season : str
        Influenza season (e.g. '2023-2024').
    country : str
        Country name (e.g. 'italia').
    week : str
        Epidemiological week as two-digit string.

    Returns
    -------
    incidence : np.ndarray
        ILI incidence time series up to the selected week.
    """
    start_year, end_year = season.split("-")

    # Weeks >= 40 belong to the first calendar year of the season
    year = start_year if int(week) >= 40 else end_year

    url = (
        f"{INFLUCAST_DATA_URL}/{season}/"
        f"{country}-{year}_{week}-ILI.csv"
    )

    #Download file
    response = requests.get(url)
    response.raise_for_status()

    # Pass content to pandas exactly as a local file
    csv_buffer = io.StringIO(response.text)

    updated_week = pd.read_csv(csv_buffer)

    incidence = updated_week["incidenza"].values

    return np.asarray(incidence)