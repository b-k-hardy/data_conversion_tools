from pathlib import Path

import h5py
import numpy as np
from pyevtk.hl import imageToVTK
from tqdm import tqdm


class PressureEstimationDataset:
    # This is a data class that I am using to structure the PPE/STE pressure fields and inputs for the Methods Paper. Some assumptions have been made
    # in order to streamline the data loading process, making this code unsuitable for general use without modification.
    # COULD use matlab to load data but it's kind of annoying... will avoid for now
    def __init__(self, name: str, dx: float, dt: int, snr: str) -> None:

        # assemble paths
        self.dx = dx
        self.dt = dt
        self.name = name
        self.snr = snr
        self.vel_dir = f"../../vwerp/UM13_vel_input/{dx}mm/{dt}ms/"
        self.seg_dir = f"../../vwerp/UM13_vel_input/{dx}mm/"
        self.results_dir = f"../../UM13_in_silico_results/{dx}mm/{dt}ms/{snr}/"

        self.vel_mask = self.load_vel_mask()
        self.inlet = self.load_inlet()
        self.outlet = self.load_outlet()

        # self.vel_dx = [1.0, 1.0, 1.0]
        # self.vel_dt = 1.0

    def load_velocity(self):
        # load timeframe 1
        # check for Nt in .res attribute
        with h5py.File(
            self.vel_dir + f"UM13_{self.dx}mm_{self.dt}ms_v_1.mat", "r"
        ) as f:
            v_pointers = f["v"][:]
            u = f[v_pointers[0, 0]]["im"][:].T
            v = f[v_pointers[1, 0]]["im"][:].T
            w = f[v_pointers[2, 0]]["im"][:].T
            self.vel_dx = np.squeeze(f[v_pointers[0, 0]]["PixDim"][:]).tolist()
            res = np.squeeze(f[v_pointers[0, 0]]["res"][:]).astype(int).tolist()
            self.vel_dt = np.squeeze(f[v_pointers[0, 0]]["dt"][:])

        Nt = res[-1]

        velocity = np.empty([3] + res)
        velocity[0, :, :, :, 0] = u
        velocity[1, :, :, :, 0] = v
        velocity[2, :, :, :, 0]
        # load rest of data in for loop

        for i in range(1, Nt):
            # open mat file
            with h5py.File(
                self.vel_dir + f"UM13_{self.dx}mm_{self.dt}ms_v_{i}.mat", "r"
            ) as f:
                # first, get pointers for velocity struct
                v_pointers = f["v"][:]

                # access the images (matlab equivalent: v{1}.im)
                velocity[0, :, :, :, i] = f[v_pointers[0, 0]]["im"][:].T
                velocity[1, :, :, :, i] = f[v_pointers[1, 0]]["im"][:].T
                velocity[2, :, :, :, i] = f[v_pointers[2, 0]]["im"][:].T

        self.velocity_data = velocity

    def load_vel_mask(self):
        with h5py.File(self.seg_dir + f"UM13_{self.dx}mm_mask.mat", "r") as f:
            vel_mask = f["mask"][:].T
        return vel_mask

    def load_inlet(self):
        with h5py.File(self.seg_dir + f"UM13_{self.dx}mm_inlet.mat", "r") as f:
            inlet_mask = f["inlet"][:].T
        return inlet_mask

    def load_outlet(self):
        with h5py.File(self.seg_dir + f"UM13_{self.dx}mm_outlet.mat", "r") as f:
            outlet_mask = f["outlet"][:].T
        return outlet_mask

    def export_velocity_to_vti(self, output_dir, mask_data=False):
        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # write mask itself and mask velocity field, if desired
        if mask_data:
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_v_mask"
            imageToVTK(out_path, spacing=self.vel_dx, cellData={"mask": self.mask})
            u *= self.mask[:, :, :, np.newaxis]
            v *= self.mask[:, :, :, np.newaxis]
            w *= self.mask[:, :, :, np.newaxis]

        # write velocity field one timestep at a time
        Nt = self.velocity_data.shape[-1]
        print(f"Exporting velocity field {self.dx} x {self.dt}")
        for t in tqdm(range(Nt)):
            u = self.velocity_data[0, :, :, :, t].copy()
            v = self.velocity_data[1, :, :, :, t].copy()
            w = self.velocity_data[2, :, :, :, t].copy()
            vel = (u, v, w)
            out_path = f"{output_dir}/UM13_{self.dx}mm_{self.dt}ms_{self.snr}_v_{t:03d}"
            imageToVTK(out_path, spacing=self.vel_dx, cellData={"Velocity": vel})


# This function converts a v7.3 mat pressure dataset to a vti group for viewing in paraview
def mat_p_to_vti(data_path, output_dir, output_filename, mask_data=False):
    mask_tol = 0.0

    with h5py.File(data_path, "r") as f:
        P = f["P"][:].T
        try:
            dx = np.squeeze(f["dx"][:]).tolist()
        except KeyError:
            print("No spatial resolution information!")

        # access mask information
        # If no mask, attempt to construct mask by summing velocity magnitude over time and excluding regions inside tolerance
        # Note that this works well for in silico data, but will probably be a bit rough for in vivo data, especially if image artifacts are present
        try:
            mask = f["mask"][:].T
        except KeyError:
            mask = np.sum(P, axis=3)
            mask = np.asarray(mask > mask_tol, dtype=float)
            print("No mask information!")

    # make sure output path exists, create directory if not
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # write mask itself and mask velocity field, if desired
    if mask_data:
        out_path = f"{output_dir}/{output_filename}_mask"
        imageToVTK(out_path, spacing=dx, cellData={"mask": mask})
        P *= mask[:, :, :, np.newaxis]

    # write pressure field one timestep at a time
    T = P.shape[3]
    for t in range(T):
        p = P[:, :, :, t]
        out_path = f"{output_dir}/{output_filename}_{t:03d}"
        imageToVTK(out_path, spacing=dx, cellData={"Relative Pressure": p})


def mat_p_err_to_vti(data_path, output_dir, output_filename, mask_data=False):
    mask_tol = 0.0

    with h5py.File(data_path, "r") as f:
        P = f["P_ERR"][:].T
        try:
            dx = np.squeeze(f["dx"][:]).tolist()
        except KeyError:
            print("No spatial resolution information!")

        # access mask information
        # If no mask, attempt to construct mask by summing velocity magnitude over time and excluding regions inside tolerance
        # Note that this works well for in silico data, but will probably be a bit rough for in vivo data, especially if image artifacts are present
        try:
            mask = f["mask"][:].T
        except KeyError:
            mask = np.sum(P, axis=3)
            mask = np.asarray(mask > mask_tol, dtype=float)
            print("No mask information!")

    # make sure output path exists, create directory if not
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # write mask itself and mask velocity field, if desired
    if mask_data:
        out_path = f"{output_dir}/{output_filename}_mask"
        imageToVTK(out_path, spacing=dx, cellData={"mask": mask})
        P *= mask[:, :, :, np.newaxis]

    # write pressure error field one timestep at a time
    T = P.shape[3]
    for t in range(T):
        p = P[:, :, :, t]
        out_path = f"{output_dir}/{output_filename}_{t:02d}"
        imageToVTK(
            out_path, spacing=dx, cellData={"Relative Pressure Absolute Error": p}
        )


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

    # it would be fun to turn this into a command line tool :)

    spatial = [1.5, 2.0, 3.0]
    temporal = [60, 40, 20]

    for dx in spatial:
        for dt in temporal:
            current_dataset = PressureEstimationDataset("current", dx, dt, "SNRinf")

            current_dataset.load_velocity()
            current_dataset.export_velocity_to_vti(
                f"../../methods_paper_vti/UM13_velocity_input_vti/{dx}mm/{dt}ms/{current_dataset.snr}"
            )

            del current_dataset


if __name__ == "__main__":
    main()
