""" This is a script that analyzes CFD data

    [INSERT LONGER EXPLANATION HERE]
"""

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib import colormaps
from scipy import stats


def load_error_mask(mask_path):

    with h5py.File(mask_path, "r") as f:
        error_mask = np.asarray(f["mask"]).T.astype(bool)

    return error_mask


def load_pressure_data(pressure_path, method, mask):

    with h5py.File(pressure_path) as f:
        p_pointer = f[f"p_{method.upper()}"][:]
        pressure = np.asarray(f[p_pointer[0, 0]]["im"]).T

    return pressure[mask].flatten()


def load_cfd_data(pressure_path, mask):

    with h5py.File(pressure_path) as f:
        pressure = np.asarray(f["P"]).T

    return pressure[mask].flatten()


def weighted_least_squares(X, Y, H):
    # doing my very own weighted least-squares with closed form solution...
    W = np.diag(H.flatten())
    B = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ Y)
    B = B.flatten()

    return B


def histogram_discrete(p_cfd, estimations, dx, dt, ax):
    # this is very much a work in progress... for now, just try out the math with the baseline cases. Once it appears to work, go ahead and try with ENTIRE dataset...

    # masking... will mask AFTER using the density array. need to flatten and then diagonalize for my weighted least-squares first.

    xr = np.array([-10.0, 20.0])
    bin_count = 20

    x = np.linspace(-9.5, 19.5, bin_count)
    y = np.linspace(-9.5, 19.5, bin_count)
    xv, yv = np.meshgrid(x, y, indexing="ij")
    X = xv.flatten()
    X = X[:, np.newaxis]
    Y = yv.flatten()
    Y = Y[:, np.newaxis]
    X_ones = np.ones_like(X)
    X = np.concatenate([X_ones, X], axis=1)

    # math shit
    H, xedges, yedges = np.histogram2d(
        p_cfd, estimations, bins=bin_count, range=[xr, xr]
    )  # at some point, will probably want to double-check to make sure I'm doing this stuff correctly

    B = weighted_least_squares(X, Y, H)

    # find the normalization factor for each column (axis=1 bc H is transposed)
    H_norm = np.maximum(1, np.sum(H, axis=1))
    H = H / H_norm[:, None]

    masked_H = np.ma.array(H, mask=(H == 0))

    cmap = colormaps["inferno_r"]
    cmap.set_bad("white", 0)  # FIXME: I want 0 to show up on colorbar??? hmmmmm

    # Regression conditional formatting
    if B[0] < 0.0:
        reg_stats = f"$y = {B[1]:.3f}x {B[0]:.3f}$"  # \n$r^2 = {reg.rvalue**2:.3f}$'
    else:
        reg_stats = f"$y = {B[1]:.3f}x + {B[0]:.3f}$"  # \n$r^2 = {reg.rvalue**2:.3f}$'

    # Plotting
    ax.text(
        0.05,
        0.95,
        reg_stats,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=16,
        color="black",
    )
    ax.plot(xr, B[0] + B[1] * xr, color="black", linewidth="3")
    ax.plot(xr, xr, color="black", linestyle="--", linewidth=3)

    # plot
    density = ax.imshow(
        masked_H.T,
        interpolation="nearest",
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        vmax=1.0,
        cmap=cmap,
    )

    # Plot formatting
    ax.set_aspect("equal")
    ax.set_ylim(bottom=xr[0], top=xr[1])
    ax.set_xlim(left=xr[0], right=xr[1])
    ax.set_title(rf"{dx[:-2]} mm $\times$ {dt[:-2]} ms", fontsize=18)

    ax.tick_params(axis="both", which="major", labelsize=16)
    # ax.legend()
    # ax.tick_params(axis='both', which='minor', labelsize=12)

    print(f"Plot {dx} x {dt} Completed")


def histogram(p_cfd, estimations, dx, dt, ax):

    # Linear regression across all points
    reg = stats.linregress(p_cfd, estimations)
    x = np.array([-10.0, 20.0])  # hard coded limits that I know look good

    # Conditional formatting for regression line equation
    if reg.intercept < 0.0:
        reg_stats = (
            f"$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"
        )
    else:
        reg_stats = (
            f"$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"
        )

    # Plotting regression line, 1:1 line, and regression line equation
    ax.text(
        0.05,
        0.95,
        reg_stats,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=16,
        color="black",
    )
    ax.plot(x, reg.intercept + reg.slope * x, color="black", linewidth="3")
    ax.plot(x, x, color="black", linestyle="--", linewidth=3)

    # Manually create 2D histogram via numpy
    H, xedges, yedges = np.histogram2d(p_cfd, estimations, bins=30, range=[x, x])

    # find the normalization factor for each column (axis=1 bc H is transposed)
    # this is what gives the percentage of each column
    H_norm = np.maximum(1, np.sum(H, axis=1))
    H = H / H_norm[:, None]

    # exclude empty bins
    masked_H = np.ma.array(H, mask=(H == 0))

    cmap = colormaps["inferno_r"]
    cmap.set_bad("white", 0)  # FIXME: Do I want 0 to show up on colorbar???

    # plot the heatmap
    density = ax.imshow(
        masked_H.T,
        interpolation="nearest",
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        vmax=1.0,
        cmap=cmap,
    )

    # Plot formatting
    ax.set_aspect("equal")
    ax.set_ylim(bottom=x[0], top=x[1])
    ax.set_xlim(left=x[0], right=x[1])
    ax.set_title(rf"{dx[:-2]} mm $\times$ {dt[:-2]} ms", fontsize=18)

    ax.tick_params(axis="both", which="major", labelsize=16)
    # ax.legend()
    # ax.tick_params(axis='both', which='minor', labelsize=12)

    # progress update
    print(f"Plot {dx} x {dt} Completed")


def multi_correlation_plot(cath_measurements, estimations: dict):

    # TODO: make this a copy of the normal correlation plot, but with a loop that goes through all the methods and draws a line for each one
    # note that there will be no blocky heatmap, just a few overlayed scatter plots with lines

    # Regression Stuff
    color_list = ["red", "blue", "green"]
    text_position = [0.75, 0.85, 0.95]
    x = np.array([-6.0, 20.0])
    fig, ax = plt.subplots(figsize=(10, 8))

    for method_name, pressure in estimations.items():

        reg = stats.linregress(cath_measurements, pressure)

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
        ax.scatter(cath_measurements, pressure, color=color)

    ax.plot(x, x, color="black", linestyle="--", linewidth=3)

    # Plot formatting
    ax.set_aspect("equal")
    ax.set_ylim(bottom=x[0], top=x[1])
    ax.set_xlim(left=x[0], right=x[1])
    ax.set_title("SNRinf 1.5mm x 40ms Correlation Plot", fontsize=22)
    ax.set_xlabel(r"$\Delta \mathregular{P_{CFD}}$ [mmHg]", fontsize=22)
    ax.set_ylabel(
        r"$\Delta \mathregular{P_{est}}$ [mmHg]", fontsize=22
    )  # honestly do not know how this is working but I'm glad it is... raw f-strings wow
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(fontsize=22)
    # ax.tick_params(axis='both', which='minor', labelsize=12)
    fig.tight_layout()

    return fig


def baseline():
    # Need to make one giant figure and then pass each axes into the plotting function
    # ALSO THERE IS AN INCREDIBLE SHITLOAD OF REDUNDANT CODE HERE -> SPLIT DATA LOADS FOR vWERP and STE into two functions and then make 3rd function for plotting, that way it's the same stuff for ste and vwerp and eventually ppe, unsteady bernoulli

    spatial = ["1.5mm", "2.0mm", "3.0mm"]
    temporal = ["20ms", "40ms", "60ms"]
    methods = ["PPE", "STE"]

    for method in methods:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=3,
            figsize=(13, 12),
            sharex=True,
            sharey=True,
            layout="constrained",
        )

        for i, dx in enumerate(spatial):
            for j, dt in enumerate(temporal):

                mask = load_error_mask(
                    f"../../UM13_in_silico_results/{dx}/{dt}/SNRinf/UM13_{dx}_{dt}_SNRinf_{method}_ERR_1.mat"
                )

                pressure = load_pressure_data(
                    f"../../UM13_in_silico_results/{dx}/{dt}/SNRinf/UM13_{dx}_{dt}_SNRinf_{method}_1.mat",
                    method,
                    mask,
                )

                # load data and get arrays for regression out
                p_cfd = load_cfd_data(
                    f"../../UM13_P_CFD/{dx}/UM13_{dx}_{dt}_shifted.mat",
                    mask,
                )

                # do plotting and regression
                histogram(p_cfd, pressure, dx, dt, axes[i, j])

        img = axes[0, 0].get_images()[0]
        cbar = fig.colorbar(img, ax=axes, format=mtick.PercentFormatter(1.0))
        cbar.set_label(label="Distribution per Column", fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        cols = [r"$\Delta \mathregular{P_{CFD}}$ [mmHg]"] * 3
        rows = [rf"$\Delta \mathregular{{P_{{{method}}}}}$ [mmHg]"] * 3

        for ax, col in zip(axes[-1], cols):
            ax.set_xlabel(col, fontsize=16)

        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, fontsize=16)

        # save big figure
        fig.savefig(
            f"../../UM13_in_silico_results/correlation_plots/UM13_{method}_inf_correlation.svg"
        )
        fig.savefig(
            f"../../UM13_in_silico_results/correlation_plots/UM13_{method}_inf_correlation.pdf"
        )


def noise(snr, n_realizations):

    spatial = ["1.5mm", "2.0mm", "3.0mm"]
    temporal = ["20ms", "40ms", "60ms"]
    methods = ["PPE", "STE"]

    for method in methods:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=3,
            figsize=(13, 12),
            sharex=True,
            sharey=True,
            layout="constrained",
        )

        for i, dx in enumerate(spatial):
            for j, dt in enumerate(temporal):

                mask = load_error_mask(
                    f"../../UM13_in_silico_results/{dx}/{dt}/SNR{snr}/{method}/UM13_{dx}_{dt}_SNR{snr}_{method}_ERR_1.mat"
                )

                pressure_arr = np.empty(shape=0)
                for n in range(1, n_realizations + 1):

                    pressure = load_pressure_data(
                        f"../../UM13_in_silico_results/{dx}/{dt}/SNR{snr}/{method}/UM13_{dx}_{dt}_SNR{snr}_{method}_{n}.mat",
                        method,
                        mask,
                    )

                    pressure_arr = np.concatenate((pressure_arr, pressure))

                # load data and get arrays for regression out
                p_cfd = load_cfd_data(
                    f"../../UM13_P_CFD/{dx}/UM13_{dx}_{dt}_shifted.mat",
                    mask,
                )

                cfd_arr = np.tile(p_cfd, n_realizations)

                if snr == "inf":
                    histogram(p_cfd, pressure, dx, dt, axes[i, j])

                else:

                    # do plotting and regression
                    histogram_discrete(cfd_arr, pressure_arr, dx, dt, axes[i, j])

        img = axes[0, 0].get_images()[0]
        cbar = fig.colorbar(img, ax=axes, format=mtick.PercentFormatter(1.0))
        cbar.set_label(label="Distribution per Column", fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        cols = [r"$\Delta \mathregular{P_{CFD}}$ [mmHg]"] * 3
        rows = [rf"$\Delta \mathregular{{P_{{{method}}}}}$ [mmHg]"] * 3

        for ax, col in zip(axes[-1], cols):
            ax.set_xlabel(col, fontsize=16)

        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, fontsize=16)

        # save big figure
        fig.savefig(
            f"../../UM13_in_silico_results/correlation_plots/UM13_{method}_SNR{snr}_correlation_test.svg"
        )
        fig.savefig(
            f"../../UM13_in_silico_results/correlation_plots/UM13_{method}_SNR{snr}_correlation_test.pdf"
        )

        plt.close(fig)


if __name__ == "__main__":
    noise("inf", 1)
    noise(10, 25)
    noise(30, 25)
