""" Module defining the InVitroDataset class.

    This data class has evolved from mat7-3_to_vti.py, which I initially used as a converter tool
    to convert between MATLAB to VTK formats. I have since expanded the functionality to include 
    a full data class with more methods with a few fixed paths to fit with the UM13 dataset for the
    methods paper. This class is not intended for general use, but can be modified for future use.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib import colormaps
from pyevtk.hl import imageToVTK
from scipy import stats
from tqdm import tqdm

PA_TO_MMHG = 0.00750061683


class InVivoDataset:
    """This is a data class that I am using to structure the PPE/STE pressure fields and inputs for the Methods Paper.
    Some assumptions have been made in order to streamline the data loading process, making this code unsuitable
    for general use without modification.COULD use matlab to load data but it's kind of annoying... will avoid for now

    # FIXME: list args, methods, attributes, etc. in docstring
    """

    def __init__(self, name: str, load: tuple = ("vel", "ste")) -> None:

        # assemble paths
        self.name = name
        self.data_dir = f"/home/bkhardy/in_vivo_analysis/{name}"
        self.inlet_tl, self.inlet_fl = self.load_inlet()
        self.outlet_tl, self.outlet_fl = self.load_outlet()

        if "vel" in load:
            self.vel_mask = self.load_vel_mask()
            self.vel_data, self.vel_dx = self.load_velocity()

        if "ste" in load:
            self.ste = self.load_pressure("STE")

        if "ppe" in load:
            self.ppe = self.load_pressure("PPE")

        if "vwerp" in load:
            self.vwerp = self.load_vwerp()

    def load_velocity(self) -> tuple[np.ndarray, list]:
        """Loads the velocity field used as input

        Returns:
            tuple[np.ndarray, list, float]: tuple containing velocity field,
            spatial resolution (as a list) and temporal resolution
        """
        # load timeframe 1 and check for resolution attribute
        with h5py.File(f"{self.data_dir}/{self.name}_velField.mat") as f:
            v_pointers = f["v"][:]
            u = f[v_pointers[0, 0]]["im"][:].T
            v = f[v_pointers[0, 1]]["im"][:].T
            w = f[v_pointers[0, 2]]["im"][:].T
            vel_dx = np.squeeze(f[v_pointers[0, 0]]["dx"][:]).tolist()
            res = np.squeeze(f[v_pointers[0, 0]]["res"][:]).astype(int).tolist()

        velocity = np.empty([3] + res)
        velocity[0, :, :, :, :] = u
        velocity[1, :, :, :, :] = v
        velocity[2, :, :, :, :] = w

        return velocity, vel_dx

    def load_pressure(self, method: str) -> dict:
        """Function to load pressure field and associated attributes for ppe or ste method.

        Args:
            method (str): PPE or STE
            realization (int): Noise realization number to load

        Returns:
            dict: Dictionary containing dx, mask and pressure field.
        """

        pres_dict = {}
        with h5py.File(
            f"{self.data_dir}/{self.name}_{method.upper()}_results.mat"
        ) as f:

            pres_dict["dx"] = (np.asarray(self.vel_dx) / 2).tolist()
            pres_dict["pressure"] = np.asarray(f[f"p_{method.upper()}"]).T
            pres_dict["dp_tl"] = np.squeeze(np.asarray(f["dp_tl"]))
            pres_dict["dp_fl"] = np.squeeze(np.asarray(f["dp_fl"]))
            pres_dict["times"] = np.squeeze(np.asarray(f["times"]))
            # FIXME: add fl results after checking STE maps

        return pres_dict

    def load_vwerp(self) -> dict:
        pres_dict = {}
        with h5py.File(f"{self.data_dir}/{self.name}_vWERP_results.mat") as f:

            pres_dict["dp_tl"] = np.squeeze(np.asarray(f["dp_tl"]))
            pres_dict["dp_fl"] = np.squeeze(np.asarray(f["dp_fl"]))
            pres_dict["times"] = np.squeeze(np.asarray(f["times"]))

        return pres_dict

    # NOTE: this format is probably a lot better than just adding shit INSIDE the functions...
    # linter definitely seems to agree with me here
    def load_vel_mask(self) -> np.ndarray:
        """Load mask used for input in STE/PPE/vWERP

        Returns:
            np.ndarray: binary mask used for velocity field
        """
        with h5py.File(f"{self.data_dir}/{self.name}_STE_mask.mat") as f:
            vel_mask = np.asarray(f["mask"]).T
        return vel_mask

    def load_inlet(self) -> np.ndarray:
        """Function to load inlet mask

        Returns:
            np.ndarray: Binary inlet mask
        """
        with h5py.File(f"{self.data_dir}/{self.name}_TL_inlet.mat") as f:
            inlet_tl = np.asarray(f["inlet"]).T
        with h5py.File(f"{self.data_dir}/{self.name}_FL_inlet.mat") as f:
            inlet_fl = np.asarray(f["inlet"]).T
        return inlet_tl, inlet_fl

    def load_outlet(self) -> np.ndarray:
        """Function to load outlet mask

        Returns:
            np.ndarray: Binary outlet mask
        """
        with h5py.File(f"{self.data_dir}/{self.name}_TL_outlet.mat") as f:
            outlet_tl = np.asarray(f["outlet"]).T
        with h5py.File(f"{self.data_dir}/{self.name}_FL_outlet.mat") as f:
            outlet_fl = np.asarray(f["outlet"]).T
        return outlet_tl, outlet_fl

    def export_velocity_to_vti(self, output_dir: str) -> None:
        """Function to export velocity fields to VTI format.

        Args:
            output_dir (str): Location to export to. DOES need trailing slash.
            mask_data (bool, optional): Whether or not to export velocity mask. Defaults to False.
        """
        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # write mask itself and mask velocity field, if desired
        out_path = f"{output_dir}/{self.name}_v_mask"
        imageToVTK(out_path, spacing=self.vel_dx, cellData={"mask": self.vel_mask})

        # write velocity field one timestep at a time
        n_timesteps = self.vel_data.shape[-1]
        print(f"Exporting velocity field {self.name} to VTI format...")
        for t in tqdm(range(n_timesteps)):

            u = self.vel_data[0, :, :, :, t].copy() * self.vel_mask

            v = self.vel_data[1, :, :, :, t].copy() * self.vel_mask

            w = self.vel_data[2, :, :, :, t].copy() * self.vel_mask

            vel = (u, v, w)
            out_path = f"{output_dir}/{self.name}_v_{t:02d}"
            imageToVTK(out_path, spacing=self.vel_dx, cellData={"Velocity": vel})

    def export_pressure_to_vti(self, output_dir: str, method: str) -> None:
        """Function to export pressure fields to VTI format.

        Args:
            output_dir (str): Directory to export pressure to. Does not need trailing slash.
            method (str): STE or PPE
            mask_data (bool, optional): Whether or not mask will be exported. Defaults to False.
        """
        # FIXME: MAKE MASK MANUALLY

        if method == "STE":
            pres_dict = self.ste
        elif method == "PPE":
            pres_dict = self.ppe
        else:
            raise ValueError("Invalid method. Must be 'STE' or 'PPE'.")

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        mask = np.array(pres_dict["pressure"][:, :, :, 0] != 0, dtype=int)

        # write mask itself
        out_path = f"{output_dir}/{self.name}_P_{method.upper()}_mask"
        imageToVTK(out_path, spacing=pres_dict["dx"], cellData={"mask": mask})

        # write pressure field one timestep at a time
        n_timesteps = int(pres_dict["pressure"].shape[-1])
        for t in tqdm(range(n_timesteps)):
            p = pres_dict["pressure"][:, :, :, t].copy()
            out_path = f"{output_dir}/{self.name}_P_{method.upper()}_{t:02d}"
            imageToVTK(
                out_path, spacing=pres_dict["dx"], cellData={"Relative Pressure": p}
            )

    def export_planes_to_vti(self, output_dir: str) -> None:
        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = f"{output_dir}/{self.name}_inlet_TL"
        imageToVTK(out_path, spacing=self.vel_dx, cellData={"inlet": self.inlet_tl})
        out_path = f"{output_dir}/{self.name}_inlet_FL"
        imageToVTK(out_path, spacing=self.vel_dx, cellData={"inlet": self.inlet_fl})

        out_path = f"{output_dir}/{self.name}_outlet_TL"
        imageToVTK(out_path, spacing=self.vel_dx, cellData={"outlet": self.outlet_tl})
        out_path = f"{output_dir}/{self.name}_outlet_FL"
        imageToVTK(out_path, spacing=self.vel_dx, cellData={"outlet": self.outlet_fl})

    def hist_plot(self):

        # FIXME: HOLY SHIT THE LABELS ARE WRONG!!!
        # FLATTEN FIRST?
        ppe_p = self.ppe["pressure"].flatten()
        ste_p = self.ste["pressure"].flatten()
        ppe_nonzero = np.nonzero(ppe_p)
        ste_nonzero = np.nonzero(ste_p)
        shared_mask = np.intersect1d(ppe_nonzero, ste_nonzero)
        ppe_p = ppe_p[shared_mask]
        ste_p = ste_p[shared_mask]

        # Regression Stuff
        reg = stats.linregress(ste_p, ppe_p)
        x = np.array([-15.0, 20.0])

        if reg.intercept < 0.0:
            reg_stats = f"$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"
        else:
            reg_stats = f"$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"

        # Plotting
        fig = plt.figure(layout="constrained", figsize=(8.5, 8))
        ax = fig.add_gridspec(top=0.75, right=1.0).subplots()
        ax.set(aspect=1)
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
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

        cmap = colormaps["inferno_r"]
        cmap.set_bad("white", 0)  # FIXME: I want 0 to show up on colorbar??? hmmmmm
        count, _, _ = ax_histx.hist(ste_p, bins=35, color="black", range=x)

        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histx.tick_params(axis="y", which="major", labelsize=16)

        # create histogram
        hist, xedges, yedges = np.histogram2d(ste_p, ppe_p, bins=35, range=[x, x])

        # find the normalization factor for each column (axis=1 bc hist is transposed)
        hist_column_sum = np.maximum(1, np.sum(hist, axis=1))
        hist = hist / hist_column_sum[:, None]

        masked_hist = np.ma.array(hist, mask=hist == 0)

        # plot
        density = ax.imshow(
            masked_hist.T,
            interpolation="nearest",
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            vmax=1.0,
            cmap=cmap,
        )
        cbar = fig.colorbar(density, ax=ax, format=mtick.PercentFormatter(1.0))
        cbar.set_label(label="Distribution per Column", fontsize=18)

        # Plot formatting
        # ax.set_aspect("equal")
        ax.set_ylim(bottom=x[0], top=x[1])
        ax.set_xlim(left=x[0], right=x[1])
        ax.set_title(
            rf"{self.name} STE vs PPE",
            fontsize=22,
        )
        ax.set_xlabel(r"$\Delta \mathregular{P_{STE}}$ [mmHg]", fontsize=22)
        ax.set_ylabel(r"$\Delta \mathregular{P_{PPE}}$ [mmHg]", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=16)
        cbar.ax.tick_params(labelsize=16)
        # ax.legend()
        # fig.tight_layout()

        return fig

    # FIXME cap data from histx!!
    def hist_plot_capped(self):
        rng = np.random.default_rng()

        # FLATTEN FIRST?
        ppe_p = self.ppe["pressure"].flatten()
        ste_p = self.ste["pressure"].flatten()
        ppe_nonzero = np.nonzero(ppe_p)
        ste_nonzero = np.nonzero(ste_p)
        shared_mask = np.intersect1d(ppe_nonzero, ste_nonzero)
        ppe_p = ppe_p[shared_mask]
        ste_p = ste_p[shared_mask]

        ppe_little_pos = ppe_p[ste_p >= -1]
        ppe_little_neg = ppe_p[ste_p <= 1]
        ppe_big = ppe_p[(np.logical_or(ste_p < -1, ste_p > 1))]

        # split ste and ppe into big vals and little vals
        ste_little_pos = ste_p[ste_p >= -1]
        ste_little_neg = ste_p[ste_p <= 1]
        ste_big = ste_p[(np.logical_or(ste_p < -1, ste_p > 1))]

        ste_hist = np.histogram(ste_big, bins=35, range=[-15, 20])
        hist_max = np.max(ste_hist[0])

        little_pos_sample = rng.integers(
            0, np.min([len(ste_little_pos), len(ppe_little_pos)]), hist_max
        )
        little_neg_sample = rng.integers(
            0, np.min([len(ste_little_neg), len(ppe_little_neg)]), hist_max
        )

        ste_big = np.concatenate(
            (
                ste_big,
                ste_little_pos[little_pos_sample],
                ste_little_neg[little_neg_sample],
            )
        )
        ppe_big = np.concatenate(
            (
                ppe_big,
                ppe_little_pos[little_pos_sample],
                ppe_little_neg[little_neg_sample],
            )
        )

        # Regression Stuff
        reg = stats.linregress(ste_big, ppe_big)
        x = np.array([-15.0, 20.0])

        if reg.intercept < 0.0:
            reg_stats = f"$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"
        else:
            reg_stats = f"$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"

        # Plotting
        fig = plt.figure(layout="constrained", figsize=(8.5, 8))
        ax = fig.add_gridspec(top=0.75, right=1.0).subplots()
        ax.set(aspect=1)
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
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

        cmap = colormaps["inferno_r"]
        cmap.set_bad("white", 0)  # FIXME: I want 0 to show up on colorbar??? hmmmmm
        count, _, _ = ax_histx.hist(ste_big, bins=35, color="black", range=x)
        count_cap = np.max([count[13], count[16]])

        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histx.tick_params(axis="y", which="major", labelsize=16)

        # create histogram
        hist, xedges, yedges = np.histogram2d(ste_big, ppe_big, bins=35, range=[x, x])

        # find the normalization factor for each column (axis=1 bc hist is transposed)
        hist_column_sum = np.maximum(1, np.sum(hist, axis=1))
        hist = hist / hist_column_sum[:, None]

        masked_hist = np.ma.array(hist, mask=hist == 0)

        # plot
        density = ax.imshow(
            masked_hist.T,
            interpolation="nearest",
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            vmax=1.0,
            cmap=cmap,
        )
        cbar = fig.colorbar(density, ax=ax, format=mtick.PercentFormatter(1.0))
        cbar.set_label(label="Distribution per Column", fontsize=18)

        # Plot formatting
        # ax.set_aspect("equal")
        ax.set_ylim(bottom=x[0], top=x[1])
        ax.set_xlim(left=x[0], right=x[1])
        ax.set_title(
            rf"{self.name} STE vs PPE",
            fontsize=22,
        )
        ax.set_xlabel(r"$\Delta \mathregular{P_{STE}}$ [mmHg]", fontsize=22)
        ax.set_ylabel(r"$\Delta \mathregular{P_{PPE}}$ [mmHg]", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=16)
        cbar.ax.tick_params(labelsize=16)
        # ax.legend()
        # fig.tight_layout()

        return fig

    # 8 patient cases, 2 drops, 3 methods.
    # all methods go in the same plot. Maybe two axes for each drop? So 8 total figs?
    # since this method will be bound to the class, we just need to call per patient
    # only need to include 3 method and 2 drops here
    def pressure_trace_compare(self):
        color_list = ["green", "blue", "red"]

        # NOTE: temporarily removing STE because I saved it incorrectly like a dummy
        method_list = ["vWERP", "STE", "PPE"]
        pressure_list = [self.vwerp, self.ste, self.ppe]
        drop_list = ["tl", "fl"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for i, drop in enumerate(drop_list):
            axes[i].plot([0, 1.1], [0, 0], color="black", linestyle="--", alpha=0.5)
            for j, method in enumerate(method_list):
                axes[i].plot(
                    pressure_list[j]["times"],
                    pressure_list[j][f"dp_{drop}"],
                    label=method,
                    color=color_list[j],
                )
            axes[i].set_title(f"{self.name} {drop.upper()} Trace")
            axes[i].set_xlabel("Time [s]")
            axes[i].set_ylabel("Pressure [mmHg]")
            axes[i].legend()
            axes[i].set_ylim(
                bottom=-10,
                top=20,
            )
        # FIXME probably add zero line?
        # FIXME: add time to .mat files for plotting here. Just use generic "timestep" for now
        fig.tight_layout()
        return fig


def multi_correlation_plot(patient_list: list[str], drop: str):

    # need to load everything first mate

    vwerp_pressure = []
    ste_pressure = []
    ppe_pressure = []

    for name in patient_list:

        patient = InVivoDataset(name, ("vel", "ppe", "ste", "vwerp"))
        vwerp_pressure.extend(patient.vwerp[f"dp_{drop}"].tolist())
        ste_pressure.extend(patient.ste[f"dp_{drop}"].tolist())
        ppe_pressure.extend(patient.ppe[f"dp_{drop}"].tolist())
        del patient

    vwerp_pressure = np.asarray(vwerp_pressure)
    ste_pressure = np.asarray(ste_pressure)
    ppe_pressure = np.asarray(ppe_pressure)
    pressure_list = [vwerp_pressure, ste_pressure, ppe_pressure]
    pressure_names = ["vWERP", "STE", "PPE"]

    # Regression Stuff
    x = np.array(
        [
            np.min(np.concatenate((vwerp_pressure, ste_pressure, ppe_pressure))),
            np.max(np.concatenate((vwerp_pressure, ste_pressure, ppe_pressure))),
        ]
    )
    # x = np.array([-12.0, 16.0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for i in range(3):
        reg = stats.linregress(pressure_list[i % 3], pressure_list[(i + 1) % 3])

        if reg.intercept < 0.0:
            reg_stats = f"$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"
        else:
            reg_stats = f"$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$"

        axes[i].plot(
            x,
            reg.intercept + reg.slope * x,
            color="black",
            linewidth="3",
        )
        axes[i].scatter(pressure_list[i % 3], pressure_list[(i + 1) % 3], s=84)

        axes[i].plot(x, x, color="black", linestyle="--", linewidth=3)

        # Plot formatting
        axes[i].set_ylim(bottom=x[0], top=x[1])
        axes[i].set_xlim(left=x[0], right=x[1])
        axes[i].set_title(
            f"{pressure_names[i % 3]} vs. {pressure_names[(i + 1) % 3]}",
            fontsize=22,
        )
        axes[i].set_xlabel(
            rf"$\Delta \mathregular{{P}}$ {pressure_names[i % 3]} [mmHg]",
            fontsize=22,
        )
        axes[i].set_ylabel(
            rf"$\Delta \mathregular{{P}}$ {pressure_names[(i + 1) % 3]} [mmHg]",
            fontsize=22,
        )
        axes[i].text(
            0.05,
            0.95,
            reg_stats,
            horizontalalignment="left",
            verticalalignment="top",
            transform=axes[i].transAxes,
            fontsize=20,
            color="black",
        )
        axes[i].tick_params(axis="both", which="major", labelsize=24)
        axes[i].set_aspect("equal")

    fig.tight_layout()

    return fig


def multi_correlation_combined_plot(patient_list: list[str]):

    # need to load everything first
    vwerp_pressure = {"dp_tl": [], "dp_fl": []}
    ste_pressure = {"dp_tl": [], "dp_fl": []}
    ppe_pressure = {"dp_tl": [], "dp_fl": []}
    drop_list = ["dp_tl", "dp_fl"]
    lumen_colors = ["tab:blue", "goldenrod"]

    for name in patient_list:
        for drop in drop_list:

            patient = InVivoDataset(name, ("vel", "ppe", "ste", "vwerp"))
            vwerp_pressure[drop].extend(patient.vwerp[drop].tolist())
            ste_pressure[drop].extend(patient.ste[drop].tolist())
            ppe_pressure[drop].extend(patient.ppe[drop].tolist())
            del patient

    for drop in drop_list:
        vwerp_pressure[drop] = np.asarray(vwerp_pressure[drop])
        ste_pressure[drop] = np.asarray(ste_pressure[drop])
        ppe_pressure[drop] = np.asarray(ppe_pressure[drop])

    pressure_list = [ste_pressure, ppe_pressure]
    pressure_names = ["STE", "PPE"]

    x = np.array([-10.0, 20.0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(2):
        for j, drop in enumerate(drop_list):
            reg = stats.linregress(vwerp_pressure[drop], pressure_list[i][drop])

            if reg.intercept < 0.0:
                reg_stats = f"$m = {reg.slope:.2f}$\n$r^2 = {reg.rvalue**2:.2f}$"
            else:
                reg_stats = f"$m = {reg.slope:.2f}$\n$r^2 = {reg.rvalue**2:.2f}$"

            axes[i].plot(
                x,
                reg.intercept + reg.slope * x,
                color=lumen_colors[j],
                linewidth=2.5,
            )
            axes[i].scatter(
                vwerp_pressure[drop],
                pressure_list[i][drop],
                s=70,
                color=lumen_colors[j],
            )

            axes[i].plot(x, x, color="black", linestyle="--", linewidth=2.5)

            axes[i].text(
                0.05,
                0.95 - j * 0.15,
                reg_stats,
                horizontalalignment="left",
                verticalalignment="top",
                transform=axes[i].transAxes,
                fontsize=20,
                color=lumen_colors[j],
            )

        axes[i].tick_params(axis="both", which="major", labelsize=24)
        axes[i].set_aspect("equal")

        # Plot formatting
        axes[i].set_ylim(bottom=x[0], top=x[1])
        axes[i].set_xlim(left=x[0], right=x[1])
        axes[i].set_title(
            f"{pressure_names[i]} vs. vWERP",
            fontsize=22,
        )
        axes[i].set_xlabel(
            r"$\Delta \mathregular{{P}}$ vWERP [mmHg]",
            fontsize=22,
        )
        axes[i].set_ylabel(
            rf"$\Delta \mathregular{{P}}$ {pressure_names[i]} [mmHg]",
            fontsize=22,
        )

    fig.tight_layout()

    return fig


def make_traces(patient_list):
    for name in patient_list:

        patient = InVivoDataset(name, ("vel", "ppe", "ste", "vwerp"))
        trace_fig = patient.pressure_trace_compare()
        trace_fig.savefig(
            f"/home/bkhardy/in_vivo_analysis/trace_comparisons/{name}_traces.pdf"
        )

        del patient


def make_histograms(patient_list):
    for name in patient_list:

        patient = InVivoDataset(name, ("vel", "ppe", "ste", "vwerp"))

        hist_fig = patient.hist_plot()
        hist_fig.savefig(
            f"/home/bkhardy/in_vivo_analysis/correlation_plots/{name}_heatmap.pdf"
        )

        del patient


def make_capped_histograms(patient_list):
    for name in patient_list:

        patient = InVivoDataset(name, ("vel", "ppe", "ste", "vwerp"))

        hist_fig = patient.hist_plot_capped()
        hist_fig.savefig(
            f"/home/bkhardy/in_vivo_analysis/correlation_plots/{name}_capped.pdf"
        )

        del patient


def make_method_comparison_plots(patient_list):
    correlation_fig_tl = multi_correlation_plot(patient_list, "tl")
    correlation_fig_tl.savefig(
        "/home/bkhardy/in_vivo_analysis/correlation_plots/tl_correlation.pdf"
    )

    correlation_fig_fl = multi_correlation_plot(patient_list, "fl")
    correlation_fig_fl.savefig(
        "/home/bkhardy/in_vivo_analysis/correlation_plots/fl_correlation.pdf"
    )


def main():
    patient_list = ["UM1", "UM2", "UM5", "UM8", "UM9", "UM11", "UM13", "UM16"]
    fig = multi_correlation_combined_plot(patient_list)
    fig.savefig(
        "/home/bkhardy/in_vivo_analysis/correlation_plots/combined_correlation.pdf"
    )
    # make_method_comparison_plots(patient_list1)
    # make_capped_histograms(patient_list)


if __name__ == "__main__":
    main()
