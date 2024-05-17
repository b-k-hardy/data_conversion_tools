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


def ste_load_noisey(data_dir_prefix, dx, dt, snr, n_realizations):

    # decide on paths based on the dt and dx combination...
    # NOTE: DID THIS LATE AT NIGHT WHILE I WAS TIRED AS FUCK... GONNA WANT TO DOUBLE CHECK EVERYTHING!!!

    cfd_path = f"{data_dir_prefix}/UM13_P_CFD/{dx}/UM13_{dx}_{dt}_shifted.mat"
    error_path = f"{data_dir_prefix}/error_masks/UM13_{dx}_error_mask_full.mat"

    # Load in cfd pressure solution
    with h5py.File(cfd_path, "r") as f:
        p_cfd = f["P"][:].T

    # need to load in error mask as well
    with h5py.File(error_path, "r") as f:
        error_mask = f["mask"][:].T.astype(bool)

    p_cfd = p_cfd[error_mask].flatten()

    ste_arr = np.empty(shape=0)
    for i in range(n_realizations):
        ste_path = (
            f"D:/UM13_in_silico_results/{dx}/{dt}/{snr}/UM13_{dx}_{dt}_{snr}_{i+1}.mat"
        )
        # Load in STE pressure estimates
        with h5py.File(ste_path, "r") as f:

            # first, get pointers for pressure struct
            p_pointers = f["p_STE"]

            p_ste = f[p_pointers[0, 0]]["im"][:].T

            p_ste = p_ste[error_mask].flatten()
            ste_arr = np.concatenate((ste_arr, p_ste))

    cfd_arr = np.tile(p_cfd, n_realizations)

    return cfd_arr, ste_arr


def hist_plot_noise(p_cfd, estimations, dx, dt, ax):
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

    # construct our weight matrix
    W = np.diag(H.flatten())

    # need to figure out X and Y... also remember I need to add a "1" function to one of the columns of X. (0th column I'm pretty sure)
    # just try out a meshgrid for now and then see what happens...

    # doing my very own weighted least-squares with closed form solution...
    B = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ Y)
    B = B.flatten()

    # find the normalization factor for each column (axis=1 bc H is transposed)
    H_norm = np.maximum(1, np.sum(H, axis=1))
    H = H / H_norm[:, None]

    masked_H = np.ma.array(H, mask=(H == 0))

    # cmap = cm.get_cmap("inferno_r").copy()
    cmap = mpl.colormaps["inferno_r"].copy()
    cmap.set_bad("white", 1e-7)  # FIXME: I want 0 to show up on colorbar??? hmmmmm

    # Regression Stuff

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


def hist_plot(p_cfd, estimations, dx, dt, ax):

    # Regression Stuff
    reg = stats.linregress(p_cfd, estimations)
    # x = np.array([np.min(p_cfd), np.max(p_cfd)])
    x = np.array([-10.0, 20.0])

    if reg.intercept < 0.0:
        reg_stats = (
            f"$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"
        )
    else:
        reg_stats = (
            f"$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"
        )

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
    ax.plot(x, reg.intercept + reg.slope * x, color="black", linewidth="3")
    ax.plot(x, x, color="black", linestyle="--", linewidth=3)

    # math shit
    H, xedges, yedges = np.histogram2d(
        p_cfd, estimations, bins=30, range=[x, x]
    )  # at some point, will probably want to double-check to make sure I'm doing this stuff correctly

    # find the normalization factor for each column (axis=1 bc H is transposed)
    H_norm = np.maximum(1, np.sum(H, axis=1))
    H = H / H_norm[:, None]

    masked_H = np.ma.array(H, mask=(H == 0))

    cmap = colormaps["inferno_r"]
    cmap.set_bad("white", 0)  # FIXME: I want 0 to show up on colorbar??? hmmmmm

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
    ax.set_ylim(bottom=x[0], top=x[1])
    ax.set_xlim(left=x[0], right=x[1])
    ax.set_title(rf"{dx[:-2]} mm $\times$ {dt[:-2]} ms", fontsize=18)

    ax.tick_params(axis="both", which="major", labelsize=16)
    # ax.legend()
    # ax.tick_params(axis='both', which='minor', labelsize=12)

    print(f"Plot {dx} x {dt} Completed")


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
                hist_plot(p_cfd, pressure, dx, dt, axes[i, j])

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

    # plt.show()


def snr10():

    data_dir_prefix = r"D:/UM13_new_analysis"

    spatial = ["1.5mm", "2.0mm", "3.0mm"]
    temporal = ["20ms", "40ms", "60ms"]

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

            if dx == "2.0mm" and dt == "20ms":
                continue

            cfd_arr, ste_arr = ste_load_noisey(data_dir_prefix, dx, dt, "SNR10", 25)

            hist_plot_noise(cfd_arr, ste_arr, dx, dt, axes[i, j])

    img = axes[0, 0].get_images()[0]
    cbar = fig.colorbar(img, ax=axes, format=mtick.PercentFormatter(1.0))
    cbar.set_label(label="Distribution per Column", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    cols = [r"$\Delta \mathregular{P_{CFD}}$ [mmHg]"] * 4
    rows = [r"$\Delta \mathregular{P_{STE}}$ [mmHg]"] * 2

    for ax, col in zip(axes[-1], cols):
        ax.set_xlabel(col, fontsize=16)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, fontsize=16)

    # save big figure
    fig.savefig(f"UM13_SNR10_correlation.svg")
    fig.savefig(f"UM13_SNR10_correlation.png", dpi=400)

    # plt.show()


def snr30():
    data_dir_prefix = r"D:/UM13/baseline_results"

    spatial = ["1.5mm", "3mm"]
    temporal = ["10ms", "20ms", "40ms", "60ms"]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(16, 7),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for i, dx in enumerate(spatial):
        for j, dt in enumerate(temporal):

            cfd_arr, ste_arr = ste_load_noisey(data_dir_prefix, dx, dt, "SNR30")

            hist_plot_noise(cfd_arr, ste_arr, dx, dt, axes[i, j])

    img = axes[0, 0].get_images()[0]
    cbar = fig.colorbar(img, ax=axes, format=mtick.PercentFormatter(1.0))
    cbar.set_label(label="Distribution per Column", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    cols = [r"$\Delta \mathregular{P_{CFD}}$ [mmHg]"] * 4
    rows = [r"$\Delta \mathregular{P_{STE}}$ [mmHg]"] * 2

    for ax, col in zip(axes[-1], cols):
        ax.set_xlabel(col, fontsize=16)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, fontsize=16)

    # save big figure
    fig.savefig(f"UM13_SNR30_correlation.svg")
    fig.savefig(f"UM13_SNR30_correlation.png", dpi=400)

    # plt.show()


if __name__ == "__main__":
    baseline()
    # snr30()
    # snr10()
