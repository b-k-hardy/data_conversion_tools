import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy import stats


def ste_load(data_dir_prefix, model):

    cath_measurements = []
    ste_estimations = []

    cath_path = f"{data_dir_prefix}/{model}/{model}_shifted_dP_array.mat"
    vwerp_path = f"{data_dir_prefix}/{model}/{model}_STE_dP_array.mat"

    # Load in catheter pressure measurements and sub-select comparison time points
    with h5py.File(cath_path, "r") as f:

        cath_time = f["cath_time"][:].flatten()[
            30:400:20
        ]  # unused, but might as well keep just in case

        # false lumen dp - PAY ATTENTION TO ORDER SO COMPARISONS ARE PAIRED CORRECTLY
        cath_measurements += f["f1_dp"][:].flatten()[30:400:20].tolist()
        cath_measurements += f["f2_dp"][:].flatten()[30:400:20].tolist()
        cath_measurements += f["f3_dp"][:].flatten()[30:400:20].tolist()
        cath_measurements += f["f4_dp"][:].flatten()[30:400:20].tolist()
        cath_measurements += f["f5_dp"][:].flatten()[30:400:20].tolist()

        # true lumen dp - Note the doubling up of the inlet-outlet dP
        cath_measurements += f["t1_dp"][:].flatten()[30:400:20].tolist()
        cath_measurements += f["t2_dp"][:].flatten()[30:400:20].tolist()
        cath_measurements += f["t3_dp"][:].flatten()[30:400:20].tolist()
        cath_measurements += f["t4_dp"][:].flatten()[30:400:20].tolist()
        cath_measurements += f["t5_dp"][:].flatten()[30:400:20].tolist()
        cath_measurements += f["o_dp"][:].flatten()[30:400:20].tolist()

    # Load in STE pressure estimates
    with h5py.File(vwerp_path, "r") as f:

        # false lumen dp
        ste_estimations += f["f1_dp"][:].flatten().tolist()
        ste_estimations += f["f2_dp"][:].flatten().tolist()
        ste_estimations += f["f3_dp"][:].flatten().tolist()
        ste_estimations += f["f4_dp"][:].flatten().tolist()
        ste_estimations += f["f5_dp"][:].flatten().tolist()

        # true lumen dp
        ste_estimations += f["t1_dp"][:].flatten().tolist()
        ste_estimations += f["t2_dp"][:].flatten().tolist()
        ste_estimations += f["t3_dp"][:].flatten().tolist()
        ste_estimations += f["t4_dp"][:].flatten().tolist()
        ste_estimations += f["t5_dp"][:].flatten().tolist()
        ste_estimations += f["o_dp"][:].flatten().tolist()

    # convert measurements to numpy array for faster regression and plotting
    cath_measurements = np.asarray(cath_measurements)
    ste_estimations = np.asarray(ste_estimations)

    return cath_measurements, ste_estimations


def hist_plot(p_cfd, estimations, model, ax):

    # STUPID ASS TEMPORARY SOLUTION
    if model == "TBAD":
        model = r"$\mathregular{TBAD_{OR}}$"
    elif model == "TBAD_ENT":
        model = r"$\mathregular{TBAD_{ENT}}$"
    elif model == "TBAD_EXT":
        model = r"$\mathregular{TBAD_{EXT}}$"

    # Regression Stuff
    reg = stats.linregress(p_cfd, estimations)
    # x = np.array([np.min(p_cfd), np.max(p_cfd)])
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
        p_cfd, estimations, bins=26, range=[x, x]
    )  # at some point, will probably want to double-check to make sure I'm doing this stuff correctly

    # find the normalization factor for each column (axis=1 bc H is transposed)
    H_norm = np.maximum(1, np.sum(H, axis=1))
    H = H / H_norm[:, None]

    masked_H = np.ma.array(H, mask=(H == 0))

    cmap = cm.get_cmap("inferno_r").copy()
    cmap.set_bad("white", -1e-7)  # FIXME: I want 0 to show up on colorbar??? hmmmmm

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
    ax.set_title(rf"{model}", fontsize=18)

    ax.tick_params(axis="both", which="major", labelsize=16)
    # ax.legend()
    # ax.tick_params(axis='both', which='minor', labelsize=12)

    print(f"Plot {model} Completed")


def original():

    # Need to load each model in separately FOR THE BASLINE SHIT
    data_dir_prefix = "../../../../../Relative Pressure Estimation/vwerp/judith"

    models = ["TBAD", "TBAD_ENT", "TBAD_EXT"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 4),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for i, model in enumerate(models):

        cath, ste = ste_load(data_dir_prefix, model)

        hist_plot(cath, ste, model, axes[i])

    img = axes[0].get_images()[0]
    cbar = fig.colorbar(img, ax=axes, format=mtick.PercentFormatter(1.0))
    cbar.set_label(label="Distribution per Column", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    # fig.suptitle('Flow Phantom SNR inf')

    cols = [r"$\Delta \mathregular{P_{cath}}$ [mmHg]"] * 3
    row = r"$\Delta \mathregular{P_{STE}}$ [mmHg]"

    for ax, col in zip(axes, cols):
        ax.set_xlabel(col, fontsize=16)

    axes[0].set_ylabel(row, fontsize=16)

    fig.savefig(f"judith_inf_correlation.svg")
    fig.savefig(f"judith_inf_correlation.pdf")
    fig.savefig(f"judith_inf_correlation.png", dpi=400)

    plt.show()


def main():
    original()


if __name__ == "__main__":
    main()
