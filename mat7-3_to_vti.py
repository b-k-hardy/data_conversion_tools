from pathlib import Path  # NOTE: this is just a reminder to switch to pathlib!!

import h5py
import numpy as np
from pyevtk.hl import imageToVTK


# This function converts a version 7.3 mat file into a vti group for viewing in paraview
def mat_v_to_vti(data_path, output_dir, output_filename, mask_data=True):
    mask_tol = 0.001

    # open mat file
    with h5py.File(data_path, "r") as f:
        # first, get pointers for velocity struct
        v_pointers = f["v"][:]

        # access the images (matlab equivalent: v{1}.im)
        u = f[v_pointers[0, 0]]["im"][:].T
        v = f[v_pointers[1, 0]]["im"][:].T
        w = f[v_pointers[2, 0]]["im"][:].T

        # access spatial resolution information (generally more important when NOT isotropic)
        try:
            dx = np.squeeze(f[v_pointers[0, 0]]["PixDim"][:]).tolist()
        except KeyError:
            dx = [0.001, 0.001, 0.001]  # defaulting to 1 mm isotropic
            print("No spatial resolution information!")

        # access mask information
        # If no mask, attempt to construct mask by summing velocity magnitude over time and excluding regions inside tolerance
        # Note that this works well for in silico data, but will probably be a bit rough for in vivo data, especially if image artifacts are present
        try:
            mask = f["mask"][:].T
        except KeyError:
            mask = np.sum(np.sqrt(u**2 + v**2 + w**2), axis=3)
            mask = np.asarray(mask > mask_tol, dtype=float)
            print("No mask information!")

    # make sure output path exists, create directory if not
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # write mask itself and mask velocity field, if desired
    if mask_data:
        out_path = f"{output_dir}/{output_filename}_mask"
        imageToVTK(out_path, spacing=dx, cellData={"mask": mask})
        u *= mask[:, :, :, np.newaxis]
        v *= mask[:, :, :, np.newaxis]
        w *= mask[:, :, :, np.newaxis]

    # write velocity field one timestep at a time
    T = u.shape[3]
    for t in range(T):
        vel = (u[:, :, :, t], v[:, :, :, t], w[:, :, :, t])
        out_path = f"{output_dir}/{output_filename}_{t:03d}"
        imageToVTK(out_path, spacing=dx, cellData={"Velocity": vel})


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
        out_path = f"{output_dir}/{output_filename}_{t:03d}"
        imageToVTK(
            out_path, spacing=dx, cellData={"Relative Pressure Absolute Error": p}
        )


# OLD CODE FOR DOING ENTIRE CONVERSION RUN-THROUGH AT ONCE
"""
def make_baseline_visualizations():
    spatial = ['0.75', '1.5', '3']
    temporal = ['10', '20', '40', '60']

    for dx in spatial:
        for dt in temporal:
            print(f"Converting {dx}mm - {dt}ms...")

            data_path_pressure = f"UM13_in_silico_results/{dx}mm/{dt}ms/baseline/UM13_{dx}mm_{dt}ms_P_STE.mat"
            data_path_error = f"UM13_in_silico_results/{dx}mm/{dt}ms/baseline/UM13_{dx}mm_{dt}ms_P_error.mat"

            pressure_output_dir = f"UM13_in_silico_results/baseline_pressure_vti/{dx}mm/{dt}ms"
            error_output_dir = f"UM13_in_silico_results/baseline_error_vti/{dx}mm/{dt}ms"

            pressure_out_filename = f"UM13_{dx}mm_{dt}ms_P_STE_baseline"     #vti is automatically appended
            error_out_filename = f"UM13_{dx}mm_{dt}ms_P_error_baseline"

            mat_p_to_vti(data_path_pressure, pressure_output_dir, pressure_out_filename)
            mat_p_err_to_vti(data_path_error, error_output_dir, error_out_filename)


def make_noise_visualizations():
    spatial = ['0.75', '1.5', '3']
    temporal = ['10', '20', '40', '60']
    noise = ['SNR10', 'SNR30']

    for dx in spatial:
        for dt in temporal:
            for snr in noise:
                # always use 25th iteration because that's the one that has the velocity field with it...
                data_path_pressure = f"UM13_in_silico_results/{dx}mm/{dt}ms/{snr}/STE_25.mat"
                data_path_error = f"UM13_in_silico_results/{dx}mm/{dt}ms/{snr}/P_error_25.mat"

                pressure_output_dir = f"UM13_in_silico_results/noise_pressure_vti/{dx}mm/{dt}ms/{snr}"
                error_output_dir = f"UM13_in_silico_results/noise_error_vti/{dx}mm/{dt}ms/{snr}"

                pressure_out_filename = f"UM13_{dx}mm_{dt}ms_{snr}_P_STE_25"     #vti is automatically appended
                error_out_filename = f"UM13_{dx}mm_{dt}ms_{snr}_P_error_25"

                mat_p_to_vti(data_path_pressure, pressure_output_dir, pressure_out_filename)
                mat_p_err_to_vti(data_path_error, error_output_dir, error_out_filename)
"""


def main():
    """
    temporal = ['10ms', '20ms', '40ms', '60ms']
    spatial = ['0.75mm', '1.5mm', '3mm']

    for dx in spatial:
        for dt in temporal:

            data_path_vel_CFD = f'../Relative Pressure Estimation/in_silico/UM13_systole/pressure_estimation/mat_files_{dx}/UM13_{dx}_{dt}.mat'

            vel_output_dir = f'D:/UM13/velocity_visualizations_updated/{dx}/{dt}'
            vel_output_filename = f'UM13_{dx}_{dt}_V'

            mat_v_to_vti(data_path_vel_CFD, vel_output_dir, vel_output_filename, mask_data=True)
    """

    noise_levels = ["SNR10", "SNR30", "SNRinf"]

    for snr in noise_levels:
        data_path_vel = f"UM13_noisey_velocity/UM13_1.5mm_10ms_{snr}_V.mat"

        vel_output_dir = f"UM13_noisey_velocity/1.5mm/10ms/{snr}"
        vel_output_filename = f"UM13_1.5mm_10ms_{snr}_V"

        mat_v_to_vti(
            data_path_vel, vel_output_dir, vel_output_filename, mask_data=False
        )


if __name__ == "__main__":
    main()
