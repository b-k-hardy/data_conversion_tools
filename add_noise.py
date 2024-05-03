import fft_downsampling as fft
import numpy as np

# NOTE: make sure to double check dimensions of mask before and after matlab/python conversion


def add_complex_noise(u, v, w, vel_mask, targetSNR, venc):

    Nx, Ny, Nz = u.shape

    # Initialize new images -- using funky dimensions to mimic the cropping process in fft_downsampling
    u_noise = np.zeros((2 * int(Nx // 2), 2 * int(Ny // 2), 2 * int(Nz // 2)))
    v_noise = np.zeros((2 * int(Nx // 2), 2 * int(Ny // 2), 2 * int(Nz // 2)))
    w_noise = np.zeros((2 * int(Nx // 2), 2 * int(Ny // 2), 2 * int(Nz // 2)))

    mag_u = np.zeros((2 * int(Nx // 2), 2 * int(Ny // 2), 2 * int(Nz // 2)))
    mag_v = np.zeros((2 * int(Nx // 2), 2 * int(Ny // 2), 2 * int(Nz // 2)))
    mag_w = np.zeros((2 * int(Nx // 2), 2 * int(Ny // 2), 2 * int(Nz // 2)))

    targetSNRdb = 10 * np.log10(targetSNR)

    mag_multiplier = (
        120  # just setting to 120 for now since I don't think it actually matters...
    )
    mag_image = vel_mask * mag_multiplier

    # DO the downsampling
    u_noise, mag_u = fft.downsample_phase_img(u, mag_image, venc, 1, targetSNRdb)
    v_noise, mag_v = fft.downsample_phase_img(v, mag_image, venc, 1, targetSNRdb)
    w_noise, mag_w = fft.downsample_phase_img(w, mag_image, venc, 1, targetSNRdb)

    return u_noise, v_noise, w_noise
