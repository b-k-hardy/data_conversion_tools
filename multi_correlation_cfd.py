""" Script to produce multi-method correlation plots for in silico plane to plane analysis

    Brandon Hardy
    09/21/2024 (AMERICAN DATE FORMAT!!!)
"""

from itertools import product
from pathlib import Path

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def dictionary_load(dict_path: str, keys: list, eng) -> np.ndarray:
    """Function to load pressure data from a dictionary file using matlab engine

    Args:
        dict_path (str): path to dictionary file
        keys (list): name of planes to extract from dictionary
        eng (_type_): initalized matlab engine

    Returns:
        np.ndarray: pressures from dictionary
    """

    pressures = []

    for key in keys:

        pressures += (
            np.asarray(eng.unpack_dictionary(dict_path, key)).flatten().tolist()
        )

    return np.array(pressures)


def multi_correlation_plot(ground_truth: np.ndarray, estimations: dict) -> plt.Figure:
    """Generates a multi-method correlation plot figure. No heatmap, just a scatter plot with regression lines.

    Args:
        ground_truth (np.ndarray): array of ground truth pressure values (e.g., CFD plane dP's)
        estimations (dict): dictionary of method names and their respective pressure estimations

    Returns:
        plt.Figure: matplotlib figure of multi-method correlation plot
    """

    # Regression Stuff
    color_list = ["red", "blue", "green"]
    text_position = [0.75, 0.85, 0.95]
    x = np.array([-10.0, 20.0])  # hard coded limits that I know look good
    fig, ax = plt.subplots(figsize=(10, 8))

    for method_name, pressure in estimations.items():

        reg = stats.linregress(ground_truth, pressure)

        if reg.intercept < 0.0:
            reg_stats = f"$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"
        else:
            reg_stats = f"$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"

        color = color_list.pop()

        ax.text(
            0.05,
            text_position.pop(),
            reg_stats,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=20,
            color=color,
        )
        ax.plot(
            x,
            reg.intercept + reg.slope * x,
            color=color,
            linewidth="3",
            label=method_name,
        )
        ax.scatter(ground_truth, pressure, color=color)

    ax.plot(x, x, color="black", linestyle="--", linewidth=3)

    # Plot formatting
    ax.set_aspect("equal")
    ax.set_ylim(bottom=x[0], top=x[1])
    ax.set_xlim(left=x[0], right=x[1])
    ax.set_title("SNRinf In Silico 1.5mm x 40 ms Correlation Plot", fontsize=22)
    ax.set_xlabel(r"$\Delta \mathregular{P_{CFD}}$ [mmHg]", fontsize=22)
    ax.set_ylabel(r"$\Delta \mathregular{P_{est}}$ [mmHg]", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(fontsize=22)
    fig.tight_layout()

    return fig


def main():
    """main function. Goal is to plot a correlation plot for each dx, dt combination with no added noise"""

    spatial = ["1.5mm", "2.0mm", "3.0mm"]
    temporal = ["20ms", "40ms", "60ms"]
    # make combinations of parameters without a ton of nested for loops
    spatiotemporal = product(spatial, temporal)

    methods = ["vWERP", "STE_dP", "PPE_dP"]
    keys = ["FL1", "FL2", "FL3", "FL4", "TL1", "TL2", "TL3", "TL4"]

    eng = matlab.engine.start_matlab()

    for dx, dt in spatiotemporal:

        cfd_path = f"../../UM13_P_CFD/{dx}/UM13_{dx}_{dt}_shifted_dP.mat"
        cfd_data = dictionary_load(cfd_path, keys, eng)
        estimate_dict = {}

        for method in methods:

            dict_path = f"../../UM13_in_silico_results/{dx}/{dt}/SNRinf/{method}/UM13_{dx}_{dt}_SNRinf_{method}_1.mat"
            estimate_dict[method] = dictionary_load(dict_path, keys, eng)

        all_fig = multi_correlation_plot(cfd_data, estimate_dict)
        fig_path = Path("../../UM13_in_silico_results/multi_correlation_plots")
        fig_path.mkdir(parents=True, exist_ok=True)
        all_fig.savefig(fig_path / f"UM13_{dx}_{dt}_SNRinf_multi_correlation_plot.pdf")
        all_fig.savefig(fig_path / f"UM13_{dx}_{dt}_SNRinf_multi_correlation_plot.svg")


if __name__ == "__main__":
    main()
