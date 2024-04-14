import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mtick
import h5py
from scipy import stats


# FIXME: NOT PARTICULARLY EFFICIENT BUT OH WELL
# LOAD data for STE comparison!! weeooo starting with 1.5 mm x 40 ms case and then going from there my boi
def ste_load(data_dir_prefix, dx, dt):

    # decide on paths based on the dt and dx combination...
    # NOTE: DID THIS LATE AT NIGHT WHILE I WAS TIRED AS FUCK... GONNA WANT TO DOUBLE CHECK EVERYTHING!!!
    cfd_path = f'{data_dir_prefix}/UM13_P_CFD/{dx}/UM13_{dx}_{dt}_shifted.mat'
    ste_path = f'{data_dir_prefix}/baseline_results/UM13_{dx}_{dt}_P_STE.mat'
    error_path = f'{data_dir_prefix}/error_masks/UM13_{dx}_error_mask_full.mat'

    # Load in catheter pressure measurements and sub-select comparison time points
    with h5py.File(cfd_path, 'r') as f:
        p_cfd = f['P'][:].T

    # Load in STE pressure estimates
    with h5py.File(ste_path, 'r') as f:
        p_ste = f['P'][:].T

    # need to load in error mask as well
    with h5py.File(error_path, 'r') as f:
        error_mask = f['mask'][:].T.astype(bool)

    #print(f'STE SHAPE: {p_ste.shape}\nCFD SHAPE: {p_cfd.shape}\n\n')

    # grab them pressure points :)
    p_ste = p_ste[error_mask]
    p_cfd = p_cfd[error_mask]

    return p_cfd.flatten(), p_ste.flatten()



def ste_load_noisey(data_dir_prefix, dx, dt, snr):

    # decide on paths based on the dt and dx combination...
    # NOTE: DID THIS LATE AT NIGHT WHILE I WAS TIRED AS FUCK... GONNA WANT TO DOUBLE CHECK EVERYTHING!!!

    # NOTE: WHOLE LOT OF TEMPORARY FIXES HERE MY FRIEND...

    if dt == '10ms':
        cfd_path = f'D:/UM13/baseline_results/CFD/UM13_-75mm_10ms_offset_shifted.mat'
    else:
        cfd_path = f'D:/UM13/baseline_results/CFD/UM13_-75mm_10ms_shifted.mat'

    error_path = f'D:/UM13/baseline_results/error_masks/UM13_{dx}_error_mask_full.mat'  # TEMPORARY FIX SIR


    # need to load in error mask as well
    with h5py.File(error_path, 'r') as f:
        error_mask = f['mask'][:].T.astype(bool)

    # Load in catheter pressure measurements and sub-select comparison time points
    with h5py.File(cfd_path, 'r') as f:
        if dt == '10ms':
            p_cfd = f['P'][:].T
        elif dt == '20ms':
            p_cfd = f['P'][1::2,:,:,:].T
        elif dt == '40ms':
            p_cfd = f['P'][2:-1:4,:,:,:].T  # last timestep exists in the 40 ms increments but wasn't solved for :)
        elif dt == '60ms':
            p_cfd = f['P'][3::6,:,:,:].T

        p_cfd = p_cfd[error_mask].flatten()

    ste_arr = np.empty(shape=0)
    for i in range(1):
        ste_path = f'D:/UM13_in_silico_results/{dx}/{dt}/{snr}/STE_{i+1}.mat'
        # Load in STE pressure estimates
        with h5py.File(ste_path, 'r') as f:
            p_ste = f['P'][:].T
            p_ste = p_ste[error_mask].flatten()
            ste_arr = np.concatenate((ste_arr, p_ste))

    cfd_arr = np.tile(p_cfd, 1)

    return cfd_arr, ste_arr


def hist_plot(p_cfd, estimations, dx, dt, ax):

    # Regression Stuff
    reg = stats.linregress(p_cfd, estimations)
    #x = np.array([np.min(p_cfd), np.max(p_cfd)])
    x = np.array([-10.0, 20.0])

    if reg.intercept < 0.0:
        reg_stats = f'$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$'
    else:
        reg_stats = f'$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$'

    # Plotting
    ax.text(0.05, 0.95, reg_stats, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=16, color='black')
    ax.plot(x, reg.intercept + reg.slope*x, color='black', linewidth='3')
    ax.plot(x, x, color='black', linestyle='--', linewidth=3)

    # math shit
    H, xedges, yedges = np.histogram2d(p_cfd, estimations, bins=30, range=[x, x])   # at some point, will probably want to double-check to make sure I'm doing this stuff correctly

    # find the normalization factor for each column (axis=1 bc H is transposed)
    H_norm = np.maximum(1, np.sum(H, axis=1))
    H = H/H_norm[:,None]

    masked_H = np.ma.array(H, mask=(H == 0))

    cmap = cm.get_cmap("inferno_r").copy()
    cmap.set_bad('white', -1e-7) # FIXME: I want 0 to show up on colorbar??? hmmmmm

    # plot
    density = ax.imshow(masked_H.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmax=1.0, cmap=cmap)

    # Plot formatting
    ax.set_aspect('equal')
    ax.set_ylim(bottom=x[0], top=x[1])
    ax.set_xlim(left=x[0], right=x[1])
    ax.set_title(fr'{dx[:-2]} mm $\times$ {dt[:-2]} ms', fontsize=18)

    ax.tick_params(axis='both', which='major', labelsize=16)
    #ax.legend()
    #ax.tick_params(axis='both', which='minor', labelsize=12)

    print(f"Plot {dx} x {dt} Completed")


def baseline():
    # Need to make one giant figure and then pass each axes into the plotting function
    # ALSO THERE IS AN INCREDIBLE SHITLOAD OF REDUNDANT CODE HERE -> SPLIT DATA LOADS FOR vWERP and STE into two functions and then make 3rd function for plotting, that way it's the same stuff for ste and vwerp and eventually ppe, unsteady bernoulli
    
    data_dir_prefix = r'D:/UM13_new_analysis'

    spatial = ['1.5mm', '2.0mm', '3.0mm']
    temporal = ['20ms', '40ms', '60ms']

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13,12), sharex=True, sharey=True, layout="constrained")

    for i, dx in enumerate(spatial):
        for j, dt in enumerate(temporal):

            # load data and get arrays for regression out
            p_cfd, p_ste = ste_load(data_dir_prefix, dx, dt)

            # do plotting and regression
            hist_plot(p_cfd, p_ste, dx, dt, axes[i,j])

    img = axes[0,0].get_images()[0]
    cbar = fig.colorbar(img, ax=axes, format=mtick.PercentFormatter(1.0))
    cbar.set_label(label='Distribution per Column', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    cols = [r'$\Delta \mathregular{P_{CFD}}$ [mmHg]']*3
    rows = [r'$\Delta \mathregular{P_{STE}}$ [mmHg]']*3

    for ax, col in zip(axes[-1], cols):
        ax.set_xlabel(col, fontsize=16)

    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, fontsize=16)

    # save big figure
    fig.savefig(f'UM13_inf_correlation_NEW.svg')
    fig.savefig(f'UM13_inf_correlation_NEW.pdf')
    fig.savefig(f'UM13_inf_correlation_NEW.png', dpi=400)

    #plt.show()


def snr10():
    data_dir_prefix = r'D:/UM13/baseline_results'

    spatial = ['1.5mm', '3mm']
    temporal = ['10ms', '20ms', '40ms', '60ms']

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,7), sharex=True, sharey=True, layout="constrained")

    for i, dx in enumerate(spatial):
        for j, dt in enumerate(temporal):

            cfd_arr, ste_arr = ste_load_noisey(data_dir_prefix, dx, dt, 'SNR10')

            hist_plot(cfd_arr, ste_arr, dx, dt, axes[i,j])

    img = axes[0,0].get_images()[0]
    cbar = fig.colorbar(img, ax=axes, format=mtick.PercentFormatter(1.0))
    cbar.set_label(label='Distribution per Column', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    cols = [r'$\Delta \mathregular{P_{CFD}}$ [mmHg]']*4
    rows = [r'$\Delta \mathregular{P_{STE}}$ [mmHg]']*2

    for ax, col in zip(axes[-1], cols):
        ax.set_xlabel(col, fontsize=16)

    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, fontsize=16)

    # save big figure

    fig.savefig(f'UM13_SNR10_correlation.svg')
    fig.savefig(f'UM13_SNR10_correlation.pdf')
    fig.savefig(f'UM13_SNR10_correlation.png', dpi=400)

    #plt.show()


def snr30():
    data_dir_prefix = r'D:/UM13/baseline_results'

    spatial = ['1.5mm', '3mm']
    temporal = ['10ms', '20ms', '40ms', '60ms']

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,7), sharex=True, sharey=True, layout="constrained")

    for i, dx in enumerate(spatial):
        for j, dt in enumerate(temporal):

            cfd_arr, ste_arr = ste_load_noisey(data_dir_prefix, dx, dt, 'SNR30')

            hist_plot(cfd_arr, ste_arr, dx, dt, axes[i,j])


    img = axes[0,0].get_images()[0]
    cbar = fig.colorbar(img, ax=axes, format=mtick.PercentFormatter(1.0))
    cbar.set_label(label='Distribution per Column', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    cols = [r'$\Delta \mathregular{P_{CFD}}$ [mmHg]']*4
    rows = [r'$\Delta \mathregular{P_{STE}}$ [mmHg]']*2

    for ax, col in zip(axes[-1], cols):
        ax.set_xlabel(col, fontsize=16)

    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, fontsize=16)

    # save big figure
    fig.savefig(f'UM13_SNR30_correlation.svg')
    fig.savefig(f'UM13_SNR30_correlation.pdf')
    fig.savefig(f'UM13_SNR30_correlation.png', dpi=400)

    #plt.show()



if __name__ == "__main__":
    baseline()
    #snr30()
    #snr10()
