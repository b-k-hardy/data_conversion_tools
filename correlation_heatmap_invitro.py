""" _summary_

    [INSERT LONGER HERE]
"""

import time
import h5py
import matlab.engine
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib import colormaps
from scipy import stats


def cath_load(data_dir_prefix, repeat: int = 1):

    cath_measurements = []

    models = ["TBAD_OR", "TBAD_ENT", "TBAD_EXT"]

    for model in models:
        cath_path = f"{data_dir_prefix}/{model}/{model}_shifted_dP_array.mat"

        # Load in catheter pressure measurements and sub-select comparison time points
        with h5py.File(cath_path, "r") as f:

            # false lumen dp - PAY ATTENTION TO ORDER SO COMPARISONS ARE PAIRED CORRECTLY
            cath_measurements += np.asarray(f["f1_dp"]).flatten()[30:400:20].tolist()
            cath_measurements += np.asarray(f["f2_dp"]).flatten()[30:400:20].tolist()
            cath_measurements += np.asarray(f["f3_dp"]).flatten()[30:400:20].tolist()
            cath_measurements += np.asarray(f["f4_dp"]).flatten()[30:400:20].tolist()
            cath_measurements += np.asarray(f["f5_dp"]).flatten()[30:400:20].tolist()
            cath_measurements += np.asarray(f["o_dp"]).flatten()[30:400:20].tolist()

            # true lumen dp - Note the doubling up of the inlet-outlet dP
            cath_measurements += np.asarray(f["t1_dp"]).flatten()[30:400:20].tolist()
            cath_measurements += np.asarray(f["t2_dp"]).flatten()[30:400:20].tolist()
            cath_measurements += np.asarray(f["t3_dp"]).flatten()[30:400:20].tolist()
            cath_measurements += np.asarray(f["t4_dp"]).flatten()[30:400:20].tolist()
            cath_measurements += np.asarray(f["t5_dp"]).flatten()[30:400:20].tolist()
            cath_measurements += np.asarray(f["o_dp"]).flatten()[30:400:20].tolist()

    cath_measurements = np.asarray(cath_measurements * repeat)

    return cath_measurements


# assemble pressure measurement points and catheter points between all models into ONE heatmap correlation plot
def vwerp_load(data_dir_prefix, eng):

    keys = [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "Outlet_FL",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "Outlet_TL",
    ]

    vwerp_estimations = []

    models = ["TBAD_OR", "TBAD_ENT", "TBAD_EXT"]
    shit_list = ["BL", "BL_ENT", "BL_EXT"]

    for i, model in enumerate(models):
        vwerp_path = f"{data_dir_prefix}/{model}/vWERP_{shit_list[i]}_SNRinf_dP.mat"

        for key in keys:
            vwerp_estimations += (
                np.asarray(eng.unpack_dictionary(vwerp_path, key, "vWERP"))
                .flatten()
                .tolist()
            )  # not sure how this indentation scheme is legal but okay

    # convert measurements to numpy array for faster regression and plotting
    vwerp_estimations = np.asarray(vwerp_estimations)

    return vwerp_estimations


def sensitivity_map_load(data_dir_prefix, method: str, snr, n_realize, eng):

    keys = [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "Outlet",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "Outlet",
    ]

    ste_estimations = []

    models = ["TBAD_OR", "TBAD_ENT", "TBAD_EXT"]
    shit_list = ["BL", "BL_ENT", "BL_EXT"]

    for i, model in enumerate(models):

        for n in range(1, n_realize + 1):  # one-based indexing for my files :)

            pressure_path = f"{data_dir_prefix}/{model}/{snr}/dP_{method}/{method}_{shit_list[i]}_{snr}_dP_{n}.mat"

            for key in keys:
                ste_estimations += (
                    np.asarray(eng.unpack_dictionary(pressure_path, key, method))
                    .flatten()
                    .tolist()
                )

    # convert measurements to numpy array for faster regression and plotting
    ste_estimations = np.asarray(ste_estimations)

    return ste_estimations


def sensitivity_vwerp_load(data_dir_prefix, snr, n_realize, eng):

    keys = [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "Outlet_FL",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "Outlet_TL",
    ]

    vwerp_estimations = []

    models = ["TBAD_OR", "TBAD_ENT", "TBAD_EXT"]
    shit_list = ["BL", "BL_ENT", "BL_EXT"]

    for i, model in enumerate(models):

        for n in range(1, n_realize + 1):  # one-based indexing for my files :)

            pressure_path = f"{data_dir_prefix}/{model}/{snr}/dP_vWERP/vWERP_{shit_list[i]}_{snr}_dP_{n}.mat"

            for key in keys:
                vwerp_estimations += (
                    np.asarray(
                        eng.unpack_dictionary(pressure_path, key, "vWERP")
                    )  # NOT SURE IF THIS IS NECESSARY
                    .flatten()
                    .tolist()
                )

    # convert measurements to numpy array for faster regression and plotting
    vwerp_estimations = np.asarray(vwerp_estimations)

    return vwerp_estimations


def single_map_load(data_dir_prefix, method: str, eng):

    keys = [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "Outlet",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "Outlet",
    ]

    ste_estimations = []

    models = ["TBAD_OR", "TBAD_ENT", "TBAD_EXT"]
    shit_list = ["BL", "BL_ENT", "BL_EXT"]

    for i, model in enumerate(models):
        pressure_path = (
            f"{data_dir_prefix}/{model}/{method}_{shit_list[i]}_SNRinf_dP.mat"
        )

        for key in keys:
            ste_estimations += (
                np.asarray(eng.unpack_dictionary(pressure_path, key, method))
                .flatten()
                .tolist()
            )  # not sure how this indentation scheme is legal but okay

    # convert measurements to numpy array for faster regression and plotting
    ste_estimations = np.asarray(ste_estimations)

    return ste_estimations


def hist_plot(cath_measurements, estimations, method):

    # Regression Stuff
    reg = stats.linregress(cath_measurements, estimations)
    x = np.array([-6.0, 20.0])

    if reg.intercept < 0.0:
        reg_stats = (
            f"$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"
        )
    else:
        reg_stats = (
            f"$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"
        )

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(
        0.05,
        0.95,
        reg_stats,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=20,
        color="black",
    )
    ax.plot(x, reg.intercept + reg.slope * x, color="black", linewidth="3")
    ax.plot(x, x, color="black", linestyle="--", linewidth=3)

    # math shit
    H, xedges, yedges = np.histogram2d(
        cath_measurements, estimations, bins=26, range=[x, x]
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
    cbar = fig.colorbar(density, ax=ax, format=mtick.PercentFormatter(1.0))
    cbar.set_label(label="Distribution per Column", fontsize=18)

    # Plot formatting
    ax.set_aspect("equal")
    ax.set_ylim(bottom=x[0], top=x[1])
    ax.set_xlim(left=x[0], right=x[1])
    ax.set_title(f"{method} SNRinf All In Vitro Models Correlation Plot", fontsize=22)
    ax.set_xlabel(r"$\Delta \mathregular{P_{cath}}$ [mmHg]", fontsize=22)
    ax.set_ylabel(
        rf"$\Delta \mathregular{{P_{{{method}}}}}$ [mmHg]", fontsize=22
    )  # honestly do not know how this is working but I'm glad it is... raw f-strings wow
    ax.tick_params(axis="both", which="major", labelsize=16)
    cbar.ax.tick_params(labelsize=16)
    # ax.legend()
    # ax.tick_params(axis='both', which='minor', labelsize=12)
    fig.tight_layout()

    return fig


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
    ax.set_title("SNRinf All In Vitro Models Correlation Plot", fontsize=22)
    ax.set_xlabel(r"$\Delta \mathregular{P_{cath}}$ [mmHg]", fontsize=22)
    ax.set_ylabel(
        r"$\Delta \mathregular{P_{est}}$ [mmHg]", fontsize=22
    )  # honestly do not know how this is working but I'm glad it is... raw f-strings wow
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(fontsize=22)
    # ax.tick_params(axis='both', which='minor', labelsize=12)
    fig.tight_layout()

    return fig


def main():
    # FIXME: the distributions look oddly similar? Might want to double-check my manual data creation process... CATH is definitely being re-used, but estimations should NOT BE... bit odd...
    # NOTE: MIGHT just want to continue doing regular correlation plots for plane-to-plane data, but definitely use this heatmap idea for the fully-spatial correlation plots?
    # ALSO THERE IS AN INCREDIBLE SHITLOAD OF REDUNDANT CODE HERE -> SPLIT DATA LOADS FOR vWERP and STE into two functions and then make 3rd function for plotting,
    # that way it's the same stuff for ste and vwerp and eventually ppe, unsteady bernoulli

    eng = matlab.engine.start_matlab()

    start_time = time.time()
    data_dir_prefix = "../../in_vitro_results"

    print("Loading data...")
    # load data and package into correct arrays
    cath_data = cath_load(data_dir_prefix)
    vwerp_estimates = vwerp_load(data_dir_prefix, eng)
    # ste_estimates = ste_load(data_dir_prefix, eng)
    ste_estimates = single_map_load(data_dir_prefix, "STE", eng)
    ppe_estimates = single_map_load(data_dir_prefix, "PPE", eng)
    # ppe_estimates = ppe_load(data_dir_prefix, eng)
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")

    estimate_dict = {
        "vWERP": vwerp_estimates,
        "STE": ste_estimates,
        "PPE": ppe_estimates,
    }

    start_time = time.time()
    print("Making SNR=inf plots...")
    # make plots
    vwerp_fig = hist_plot(
        cath_data, vwerp_estimates, "vWERP"
    )  # FIXME: having a lot of trouble getting the "v" to italicize properly...
    ste_fig = hist_plot(cath_data, ste_estimates, "STE")
    ppe_fig = hist_plot(cath_data, ppe_estimates, "PPE")
    all_fig = multi_correlation_plot(cath_data, estimate_dict)
    plot_time = time.time() - start_time
    print(f"SNR=inf plots made in {plot_time:.2f} seconds")

    start_time = time.time()
    print("Making SNR=10 plots...")
    ste_total_fig = hist_plot(
        cath_load(data_dir_prefix, 50),
        sensitivity_map_load(data_dir_prefix, "STE", "SNR10", 50, eng),
        "STE",
    )
    ppe_total_fig = hist_plot(
        cath_load(data_dir_prefix, 50),
        sensitivity_map_load(data_dir_prefix, "PPE", "SNR10", 50, eng),
        "PPE",
    )
    vwerp_total_fig = hist_plot(
        cath_load(data_dir_prefix, 50),
        sensitivity_vwerp_load(data_dir_prefix, "SNR10", 50, eng),
        "vWERP",
    )
    print(f"SNR=10 plots made in {time.time() - start_time:.2f} seconds")

    print("Saving plots...")
    # save figures
    vwerp_fig.savefig(f"{data_dir_prefix}/correlation_plots/vwerp_correlation.svg")
    ste_fig.savefig(f"{data_dir_prefix}/correlation_plots/ste_correlation.svg")
    ppe_fig.savefig(f"{data_dir_prefix}/correlation_plots/ppe_correlation.svg")
    all_fig.savefig(f"{data_dir_prefix}/correlation_plots/all_correlation.svg")
    ste_total_fig.savefig(
        f"{data_dir_prefix}/correlation_plots/ste_total_correlation_snr10.svg"
    )
    ppe_total_fig.savefig(
        f"{data_dir_prefix}/correlation_plots/ppe_total_correlation_snr10.svg"
    )
    vwerp_total_fig.savefig(
        f"{data_dir_prefix}/correlation_plots/vwerp_total_correlation_snr10.svg"
    )

    vwerp_fig.savefig(f"{data_dir_prefix}/correlation_plots/vwerp_correlation.pdf")
    ste_fig.savefig(f"{data_dir_prefix}/correlation_plots/ste_correlation.pdf")
    ppe_fig.savefig(f"{data_dir_prefix}/correlation_plots/ppe_correlation.pdf")
    all_fig.savefig(f"{data_dir_prefix}/correlation_plots/all_correlation.pdf")
    ste_total_fig.savefig(
        f"{data_dir_prefix}/correlation_plots/ste_total_correlation_snr10.pdf"
    )
    ppe_total_fig.savefig(
        f"{data_dir_prefix}/correlation_plots/ppe_total_correlation_snr10.pdf"
    )
    vwerp_total_fig.savefig(
        f"{data_dir_prefix}/correlation_plots/vwerp_total_correlation_snr10.pdf"
    )

    plt.show()


if __name__ == "__main__":
    main()
