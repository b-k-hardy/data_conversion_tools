""" Module defining the PressureEstimationDataset class.

    This data class has evolved from mat7-3_to_vti.py, which I initially used as a converter tool
    to convert between MATLAB to VTK formats. I have since expanded the functionality to include 
    a full data class with more methods with a few fixed paths to fit with the UM13 dataset for the
    methods paper. This class is not intended for general use, but can be modified for future use.
"""

from pathlib import Path

import h5py
import numpy as np
from pyevtk.hl import imageToVTK
from tqdm import tqdm

# FIXME: I might want to split this up further? Maybe separate classes that are linked somehow? Not sure...
PA_TO_MMHG = 0.00750061683


class PressureEstimationDataset:
    """This is a data class that I am using to structure the PPE/STE pressure fields and inputs for the Methods Paper.
    Some assumptions have been made in order to streamline the data loading process, making this code unsuitable
    for general use without modification.COULD use matlab to load data but it's kind of annoying... will avoid for now

    # FIXME: list args, methods, attributes, etc. in docstring
    """

    def __init__(
        self, dx: float, dt: int, snr: str, load: tuple = ("vel", "ste")
    ) -> None:

        # assemble paths
        self.dx = dx
        self.dt = dt
        self.snr = snr
        self.vel_dir = f"/home/bkhardy/vwerp/UM13_vel_input/{dx}mm/{dt}ms/"
        self.seg_dir = f"/home/bkhardy/vwerp/UM13_vel_input/{dx}mm/"
        self.results_dir = f"/home/bkhardy/UM13_in_silico_results/{dx}mm/{dt}ms/{snr}/"
        self.cfd_dir = f"/home/bkhardy/UM13_P_CFD/{dx}mm/"

        if "vel" in load:
            self.vel_mask = self.load_vel_mask()
            self.velocity_data, self.vel_dx, self.vel_dt = self.load_velocity()

        if "ste" in load:
            self.ste = self.load_pressure("STE", 1)

        if "ste_err" in load:
            self.ste_err = self.load_error("STE", 1)

        if "ppe" in load:
            self.ppe = self.load_pressure("PPE", 1)

        if "ppe_err" in load:
            self.ppe_err = self.load_error("PPE", 1)

        if "cfd" in load:
            self.cfd = self.load_cfd_pressure()
        # self.inlet = self.load_inlet()
        # self.outlet = self.load_outlet()

    def load_velocity(self) -> tuple[np.ndarray, list, float]:
        """Loads the velocity field used as input

        Returns:
            tuple[np.ndarray, list, float]: tuple containing velocity field,
            spatial resolution (as a list) and temporal resolution
        """
        # load timeframe 1 and check for resolution attribute
        with h5py.File(f"{self.vel_dir}UM13_{self.dx}mm_{self.dt}ms_v_1.mat") as f:
            v_pointers = f["v"][:]
            u = f[v_pointers[0, 0]]["im"][:].T
            v = f[v_pointers[1, 0]]["im"][:].T
            w = f[v_pointers[2, 0]]["im"][:].T
            vel_dx = np.squeeze(f[v_pointers[0, 0]]["PixDim"][:]).tolist()
            res = np.squeeze(f[v_pointers[0, 0]]["res"][:]).astype(int).tolist()
            vel_dt = float(np.squeeze(f[v_pointers[0, 0]]["dt"][:]))

        n_timesteps = res[-1]

        velocity = np.empty([3] + res)
        velocity[0, :, :, :, 0] = u
        velocity[1, :, :, :, 0] = v
        velocity[2, :, :, :, 0] = w
        # load rest of data in for loop

        for i in range(1, n_timesteps):
            # open mat file
            with h5py.File(
                f"{self.vel_dir}UM13_{self.dx}mm_{self.dt}ms_v_{i}.mat"
            ) as f:
                # first, get pointers for velocity struct
                v_pointers = f["v"][:]

                # access the images (matlab equivalent: v{1}.im)
                velocity[0, :, :, :, i] = f[v_pointers[0, 0]]["im"][:].T
                velocity[1, :, :, :, i] = f[v_pointers[1, 0]]["im"][:].T
                velocity[2, :, :, :, i] = f[v_pointers[2, 0]]["im"][:].T

        return velocity, vel_dx, vel_dt

    def load_pressure(self, method: str, realization: int) -> dict:
        """Function to load pressure field and associated attributes for ppe or ste method.

        Args:
            method (str): PPE or STE
            realization (int): Noise realization number to load

        Returns:
            dict: Dictionary containing dx, mask and pressure field.
        """

        pres_dict = {}
        with h5py.File(
            f"{self.results_dir}UM13_{self.dx}mm_{self.dt}ms_{self.snr}_{method}_{realization}.mat",
        ) as file:

            p_pointer = file[f"p_{method.upper()}"][:]
            pres_dict["dx"] = np.squeeze(file[p_pointer[0, 0]]["dx"][:]).tolist()
            pres_dict["mask"] = file[p_pointer[0, 0]]["mask"][:].T
            pres_dict["pressure"] = file[p_pointer[0, 0]]["im"][:].T

        return pres_dict

    def load_error(self, method: str, realization: int) -> dict:
        """Function to load error field for ppe or ste method.

        Args:
            method (str): STE or PPE.
            iteration (int): Which noise realization to load

        Returns:
            dict: Dictionary containing dx, mask and error field.
        """

        err_dict = {}
        with h5py.File(
            f"{self.results_dir}UM13_{self.dx}mm_{self.dt}ms_{self.snr}_{method}_ERR_{realization}.mat"
        ) as file:

            err_dict["dx"] = np.squeeze(file["dx"][:]).tolist()
            err_dict["mask"] = np.asarray(file["mask"]).T
            err_dict["err"] = np.asarray(file["P_ERR"]).T

        return err_dict

    # NOTE: this format is probably a lot better than just adding shit INSIDE the functions...
    # linter definitely seems to agree with me here
    def load_vel_mask(self) -> np.ndarray:
        """Load mask used for input in STE/PPE/vWERP

        Returns:
            np.ndarray: binary mask used for velocity field
        """
        with h5py.File(f"{self.seg_dir}UM13_{self.dx}mm_mask.mat") as f:
            vel_mask = np.asarray(f["mask"]).T
        return vel_mask

    def load_inlet(self) -> np.ndarray:
        """Function to load inlet mask

        Returns:
            np.ndarray: Binary inlet mask
        """
        with h5py.File(f"{self.seg_dir}UM13_{self.dx}mm_inlet.mat") as f:
            inlet_mask = np.asarray(f["inlet"]).T
        return inlet_mask

    def load_outlet(self) -> np.ndarray:
        """Function to load outlet mask

        Returns:
            np.ndarray: Binary outlet mask
        """
        with h5py.File(f"{self.seg_dir}UM13_{self.dx}mm_outlet.mat") as f:
            outlet_mask = np.asarray(f["outlet"]).T
        return outlet_mask

    def export_velocity_to_vti(self, output_dir: str, mask_data: bool = False) -> None:
        """Function to export velocity fields to VTI format.

        Args:
            output_dir (str): Location to export to. DOES need trailing slash.
            mask_data (bool, optional): Whether or not to export velocity mask. Defaults to False.
        """
        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # write mask itself and mask velocity field, if desired
        if mask_data:
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_v_mask"
            imageToVTK(out_path, spacing=self.vel_dx, cellData={"mask": self.vel_mask})

        # write velocity field one timestep at a time
        n_timesteps = self.velocity_data.shape[-1]
        print(f"Exporting velocity field {self.dx} x {self.dt}")
        for t in tqdm(range(n_timesteps)):

            u = (
                self.velocity_data[0, :, :, :, t].copy() * self.vel_mask
                if mask_data
                else self.velocity_data[0, :, :, :, t].copy()
            )
            v = (
                self.velocity_data[1, :, :, :, t].copy() * self.vel_mask
                if mask_data
                else self.velocity_data[1, :, :, :, t].copy()
            )
            w = (
                self.velocity_data[2, :, :, :, t].copy() * self.vel_mask
                if mask_data
                else self.velocity_data[2, :, :, :, t].copy()
            )

            vel = (u, v, w)
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_{self.snr}_v_{t:02d}"
            imageToVTK(out_path, spacing=self.vel_dx, cellData={"Velocity": vel})

    def export_pressure_to_vti(
        self, output_dir: str, method: str, mask_data=False
    ) -> None:
        """Function to export pressure fields to VTI format.

        Args:
            output_dir (str): Directory to export pressure to. Does not need trailing slash.
            method (str): STE or PPE
            mask_data (bool, optional): Whether or not mask will be exported. Defaults to False.
        """

        if method == "STE":
            pres_dict = self.ste
        elif method == "PPE":
            pres_dict = self.ppe

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # write mask itself
        if mask_data:
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_{self.snr}_P_{method.upper()}_mask"
            imageToVTK(
                out_path, spacing=pres_dict["dx"], cellData={"mask": pres_dict["mask"]}
            )

        # write pressure field one timestep at a time
        n_timesteps = int(pres_dict["pressure"].shape[-1])
        for t in tqdm(range(n_timesteps)):
            p = pres_dict["pressure"][:, :, :, t].copy()
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_{self.snr}_P_{method.upper()}_{t:02d}"
            imageToVTK(
                out_path, spacing=pres_dict["dx"], cellData={"Relative Pressure": p}
            )

    def export_error_to_vti(
        self, output_dir: str, method: str, mask_data=False
    ) -> None:
        """Function to export error fields to VTI format.

        Args:
            output_dir (str): Directory to export pressure to. Does not need trailing slash.
            method (str): STE or PPE
            mask_data (bool, optional): Whether or not to export error mask. Defaults to False.
        """

        if method == "STE":
            err_dict = self.ste_err
        elif method == "PPE":
            err_dict = self.ppe_err

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # write mask itself
        if mask_data:
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_{self.snr}_P_{method.upper()}_ERR_mask"
            imageToVTK(
                out_path, spacing=err_dict["dx"], cellData={"mask": err_dict["mask"]}
            )

        # write pressure error field one timestep at a time
        n_timesteps = int(err_dict["err"].shape[-1])
        for t in tqdm(range(n_timesteps)):
            p = err_dict["err"][:, :, :, t].copy()
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_{self.snr}_P_{method.upper()}_ERR_{t:02d}"
            imageToVTK(
                out_path,
                spacing=err_dict["dx"],
                cellData={"Relative Pressure Absolute Error": p},
            )

    def load_cfd_pressure(self) -> dict:
        """Function to load pressure field and associated attributes for cfd since they have different data structures.

        Returns:
            dict: Dictionary containing dx, mask and pressure field.
        """

        pres_dict = {}
        with h5py.File(
            f"{self.cfd_dir}UM13_{self.dx}mm_{self.dt}ms_shifted.mat",
        ) as file:

            pres_dict["dx"] = np.squeeze(file["dx"][:]).tolist()
            pres_dict["mask"] = np.asarray(file["mask"]).T
            pres_dict["pressure"] = np.asarray(file["P"]).T

        return pres_dict

    def export_cfd_to_vti(self, output_dir: str, mask_data: bool = False) -> None:
        """Function to export cfd pressure fields to VTI format.

        Args:
            output_dir (str): Directory to export pressure to. Does not need trailing slash.
            mask_data (bool, optional): Whether or not mask will be exported. Defaults to False.
        """

        pres_dict = self.cfd

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # write mask itself
        if mask_data:
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_P_CFD_mask"
            imageToVTK(
                out_path, spacing=pres_dict["dx"], cellData={"mask": pres_dict["mask"]}
            )

        # write pressure field one timestep at a time
        n_timesteps = int(pres_dict["pressure"].shape[-1])
        for t in tqdm(range(n_timesteps)):
            p = pres_dict["pressure"][:, :, :, t].copy()
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_P_CFD_{t:02d}"
            imageToVTK(
                out_path, spacing=pres_dict["dx"], cellData={"Relative Pressure": p}
            )


def invitro_to_vti(data_path: str, output_dir: str, output_name: str) -> None:
    """speedy function to both load and convert vti data... if I import on own does the top stuff load idk

    Args:
        data_path (str): _description_
        output_dir (str): _description_
        output_name (str): _description_
    """

    with h5py.File(data_path) as file:
        pressure = np.asarray(
            file["P"]
        ).T  # * PA_TO_MMHG # ONLY STE IS IN PASCALS DANG  # convert to mmHg because I saved in Pascals like a dummy

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_timesteps = int(pressure.shape[-1])
    for t in tqdm(range(n_timesteps)):
        p = pressure[:, :, :, t].copy()
        out_path = f"{output_dir}/{output_name}_{t:02d}"
        imageToVTK(
            out_path,
            spacing=(0.0015 / 2, 0.0015 / 2, 0.0015 / 2),
            cellData={"Relative Pressure": p},
        )


def mag_to_vti(
    data_path: str, output_dir: str, output_name: str, model_name: str
) -> None:
    """Speedy function to fix the dumbass mag images I was provided

    Args:
        data_path (str): _description_
        output_dir (str): _description_
        output_name (str): _description_
        model_name (str): TBAD_OR, TBAD_EXT, TBAD_ENT
    """
    with h5py.File(data_path) as file:
        mag = np.asarray(file[model_name]).T

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    out_path = f"{output_dir}/{output_name}"
    imageToVTK(out_path, spacing=(0.0015, 0.0015, 0.0015), cellData={"Magnitude": mag})


def flow_to_vti(data_path: str, output_dir: str, output_name: str):

    with h5py.File(data_path) as f:
        v_pointers = f["v"][:]
        u = f[v_pointers[0, 0]]["im"][:].T
        v = f[v_pointers[1, 0]]["im"][:].T
        w = f[v_pointers[2, 0]]["im"][:].T
        vel_dx = np.squeeze(f[v_pointers[0, 0]]["dx"][:]).tolist()

        mask = f["mask"][:].T

    n_timesteps = u.shape[-1]

    # make sure output path exists, create directory if not
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for t in tqdm(range(n_timesteps)):

        u_t = u[:, :, :, t].copy() * mask
        v_t = v[:, :, :, t].copy() * mask
        w_t = w[:, :, :, t].copy() * mask

        vel = (u_t, v_t, w_t)
        out_path = f"{output_dir}/{output_name}_{t:02d}"
        imageToVTK(out_path, spacing=vel_dx, cellData={"Velocity": vel})


def export_baseline_velocity(dx_list=(1.5, 2.0, 3.0), dt_list=(60, 40, 20)) -> None:
    """Function to automate the export of baseline velocity fields to VTI format.

    Args:
        dx_list (tuple, optional): List of spatial resolutions to convert. Defaults to (1.5, 2.0, 3.0).
        dt_list (tuple, optional): List of temporal resolutions to convert. Defaults to (60, 40, 20).
    """
    for dx in dx_list:
        for dt in dt_list:
            current_dataset = PressureEstimationDataset(dx, dt, "SNRinf", load="vel")

            current_dataset.export_velocity_to_vti(
                f"../../methods_paper_vti/UM13_velocity_input_vti/{dx}mm/{dt}ms/{current_dataset.snr}"
            )

            del current_dataset


def export_results():
    dx_list = [1.5, 2.0, 3.0]
    dt_list = [60, 40, 20]
    noise_list = ["SNRinf", "SNR30", "SNR10"]

    for dx in dx_list:
        for dt in dt_list:
            for snr in noise_list:
                current_dataset = PressureEstimationDataset(
                    dx, dt, snr, load=("ppe", "ppe_err")
                )

                current_dataset.export_pressure_to_vti(
                    f"../../methods_paper_vti/UM13_pressure_output_vti/{dx}mm/{dt}ms/{current_dataset.snr}/PPE",
                    "PPE",
                )
                current_dataset.export_error_to_vti(
                    f"../../methods_paper_vti/UM13_error_output_vti/{dx}mm/{dt}ms/{current_dataset.snr}/STE",
                    "PPE",
                )

                del current_dataset

    # it would be fun to turn this into a command line tool :)


def main():
    """main method."""

    dx_list = [1.5, 2.0, 3.0]
    dt_list = [60, 40, 20]
    snr = "SNRinf"

    for dx in dx_list:
        for dt in dt_list:
            current_dataset = PressureEstimationDataset(dx, dt, snr, load="cfd")

            current_dataset.export_cfd_to_vti(
                f"/home/bkhardy/methods_paper_vti/UM13_pressure_cfd_vti/{dx}mm/{dt}ms"
            )

            del current_dataset


if __name__ == "__main__":
    main()
