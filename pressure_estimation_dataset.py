""" Module defining the PressureEstimationDataset class.

    This data class has evolved from mat7-3_to_vti.py, which I initially used as a converter tool
    to convert between MATLAB to VTK formats. I have since expanded the functionality to include 
    a full data class with more methods with a few fixed paths to fit with the UM13 dataset for the
    methods paper. This class is not intended for general use, but can be modified for future use.
"""

from pathlib import Path

import h5py
import numpy as np
from add_noise import add_complex_noise
from pyevtk.hl import imageToVTK
from tqdm import tqdm


# FIXME: I might want to split this up further? Maybe separate classes that are linked somehow? Not sure...
class PressureEstimationDataset:
    # This is a data class that I am using to structure the PPE/STE pressure fields and inputs for the Methods Paper. Some assumptions have been made
    # in order to streamline the data loading process, making this code unsuitable for general use without modification.
    # COULD use matlab to load data but it's kind of annoying... will avoid for now
    def __init__(
        self, dx: float, dt: int, snr: str, load: tuple = ("err", "vel", "pres")
    ) -> None:

        # assemble paths
        self.dx = dx
        self.dt = dt
        self.snr = snr
        self.vel_dir = f"../../vwerp/UM13_vel_input/{dx}mm/{dt}ms/"
        self.seg_dir = f"../../vwerp/UM13_vel_input/{dx}mm/"
        self.results_dir = f"../../UM13_in_silico_results/{dx}mm/{dt}ms/{snr}/"

        if "vel" in load:
            self.vel_mask = self.load_vel_mask()
            self.velocity_data, self.vel_dx, self.vel_dt = self.load_velocity()
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

    def load_pressure(self, method: str, iteration: int):
        with h5py.File(
            f"{self.results_dir}UM13_{self.dx}mm_{self.dt}ms_{self.snr}_{method}_{iteration}.mat",
            "r",
        ) as file:

            p_pointer = file[f"p_{method.upper()}"][:]
            self.pres_dx = np.squeeze(file[p_pointer[0, 0]]["dx"][:]).tolist()
            self.pres_mask = file[p_pointer[0, 0]]["mask"][:].T
            if method.upper() == "STE":
                self.ste_result = file[p_pointer[0, 0]]["im"][:].T
            elif method.upper() == "PPE":
                self.ppe_result = file[p_pointer[0, 0]]["im"][:].T

    def load_error(self, method: str, realization: int) -> None:
        """Function to load error field for ppe or ste method.

        Args:
            method (str): STE or PPE.
            iteration (int): Which noise realization to load
        """

        with h5py.File(
            f"{self.results_dir}UM13_{self.dx}mm_{self.dt}ms_{self.snr}_{method}_ERR_{realization}.mat"
        ) as file:

            self.err_dx = np.squeeze(file["dx"][:]).tolist()
            self.err_mask = np.asarray(file["mask"]).T
            if method.upper() == "STE":
                self.ste_err = np.asarray(file["P_ERR"]).T
            elif method.upper() == "PPE":
                self.ppe_err = np.asarray(file["P_ERR"]).T

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

    def export_velocity_to_vti(self, output_dir, mask_data=False) -> None:
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

            # DINGUS YOU MASKED AND THEN ADDED NOISE...
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

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # write mask itself
        if mask_data:
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_{self.snr}_P_{method.upper()}_mask"
            imageToVTK(
                out_path, spacing=self.pres_dx, cellData={"mask": self.pres_mask}
            )

        # write pressure field one timestep at a time
        # FIXME: I'M CURRENTLY FORCING STE
        n_timesteps = self.ste_result.shape[-1]
        for t in tqdm(range(n_timesteps)):
            p = self.ste_result[:, :, :, t].copy()
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_{self.snr}_P_{method.upper()}_{t:02d}"
            imageToVTK(
                out_path, spacing=self.pres_dx, cellData={"Relative Pressure": p}
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
        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # write mask itself
        if mask_data:
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_{self.snr}_P_{method.upper()}_ERR_mask"
            imageToVTK(out_path, spacing=self.err_dx, cellData={"mask": self.err_mask})

        # write pressure error field one timestep at a time
        # FIXME: forcing STE here too
        n_timesteps = self.ste_err.shape[-1]
        for t in tqdm(range(n_timesteps)):
            p = self.ste_err[:, :, :, t].copy()
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_{self.snr}_P_{method.upper()}_ERR_{t:02d}"
            imageToVTK(
                out_path,
                spacing=self.err_dx,
                cellData={"Relative Pressure Absolute Error": p},
            )


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
                f"../../methods_paper_vti/UM13_velocity_input_vti/{dx}mm/{dt}ms/{current_dataset.snr}",
                mask_data=True,
            )

            del current_dataset


def main():
    """
    noise_levels = ["SNR10", "SNR30", "SNRinf"]

    for snr in noise_levels:
        data_path_vel = f"UM13_noisey_velocity/UM13_1.5mm_10ms_{snr}_V.mat"

        vel_output_dir = f"UM13_noisey_velocity/1.5mm/10ms/{snr}"
        vel_output_filename = f"UM13_1.5mm_10ms_{snr}_V"

        mat_v_to_vti(
            data_path_vel, vel_output_dir, vel_output_filename, mask_data=False
        )
    """
    export_baseline_velocity(dx_list=[1.5], dt_list=[20])

    """
    dx_list = [1.5, 2.0, 3.0]
    dt_list = [60, 40, 20]

    for dx in dx_list:
        for dt in dt_list:
            current_dataset = PressureEstimationDataset("current", dx, dt, "SNRinf")

            current_dataset.load_error("STE", 1)
            current_dataset.export_error_to_vti(
                f"../../methods_paper_vti/UM13_error_output_vti/{dx}mm/{dt}ms/{current_dataset.snr}/STE",
                "STE",
            )
            del current_dataset  
    """

    # it would be fun to turn this into a command line tool :)


if __name__ == "__main__":
    main()
