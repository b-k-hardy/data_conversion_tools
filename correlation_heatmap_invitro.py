import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mtick
import h5py
from scipy import stats


# assemble pressure measurement points and catheter points between all models into ONE heatmap correlation plot
def vwerp_load(data_dir_prefix):

    cath_measurements = []
    vwerp_estimations = []

    models = ['TBAD', 'TBAD_ENT', 'TBAD_EXT']

    for model in models:

        cath_path = f'{data_dir_prefix}/{model}/{model}_shifted_dP_array.mat'
        vwerp_path = f'{data_dir_prefix}/{model}/{model}_vWERP_dP_array.mat'

        # Load in catheter pressure measurements and sub-select comparison time points
        with h5py.File(cath_path, 'r') as f:

            cath_time = f['cath_time'][:].flatten()[30:400:20] # unused, but might as well keep just in case

            # false lumen dp - PAY ATTENTION TO ORDER SO COMPARISONS ARE PAIRED CORRECTLY
            cath_measurements += f['f1_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['f2_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['f3_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['f4_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['f5_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['o_dp'][:].flatten()[30:400:20].tolist()

            # true lumen dp - Note the doubling up of the inlet-outlet dP
            cath_measurements += f['t1_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['t2_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['t3_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['t4_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['t5_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['o_dp'][:].flatten()[30:400:20].tolist()

        # Load in vWERP pressure estimates
        with h5py.File(vwerp_path, 'r') as f:
            # NOTE: currently including both of the outlet measurements. Wonder if it would be more "correct" to include both but weight them by half?
            
            # false lumen dp
            vwerp_estimations += f['f1_dp'][:].flatten().tolist()
            vwerp_estimations += f['f2_dp'][:].flatten().tolist()
            vwerp_estimations += f['f3_dp'][:].flatten().tolist()
            vwerp_estimations += f['f4_dp'][:].flatten().tolist()
            vwerp_estimations += f['f5_dp'][:].flatten().tolist()
            vwerp_estimations += f['of_dp'][:].flatten().tolist()

            # true lumen dp
            vwerp_estimations += f['t1_dp'][:].flatten().tolist()
            vwerp_estimations += f['t2_dp'][:].flatten().tolist()
            vwerp_estimations += f['t3_dp'][:].flatten().tolist()
            vwerp_estimations += f['t4_dp'][:].flatten().tolist()
            vwerp_estimations += f['t5_dp'][:].flatten().tolist()
            vwerp_estimations += f['ot_dp'][:].flatten().tolist()

    # convert measurements to numpy array for faster regression and plotting
    cath_measurements = np.asarray(cath_measurements)
    vwerp_estimations = np.asarray(vwerp_estimations)

    return cath_measurements, vwerp_estimations


def ste_load(data_dir_prefix):

    cath_measurements = []
    ste_estimations = []

    models = ['TBAD', 'TBAD_ENT', 'TBAD_EXT']

    for model in models:

        cath_path = f'{data_dir_prefix}/{model}/{model}_shifted_dP_array.mat'
        vwerp_path = f'{data_dir_prefix}/{model}/{model}_STE_dP_array.mat'

        # Load in catheter pressure measurements and sub-select comparison time points
        with h5py.File(cath_path, 'r') as f:

            cath_time = f['cath_time'][:].flatten()[30:400:20] # unused, but might as well keep just in case

            # false lumen dp - PAY ATTENTION TO ORDER SO COMPARISONS ARE PAIRED CORRECTLY
            cath_measurements += f['f1_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['f2_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['f3_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['f4_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['f5_dp'][:].flatten()[30:400:20].tolist()

            # true lumen dp - Note the doubling up of the inlet-outlet dP
            cath_measurements += f['t1_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['t2_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['t3_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['t4_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['t5_dp'][:].flatten()[30:400:20].tolist()
            cath_measurements += f['o_dp'][:].flatten()[30:400:20].tolist()

        # Load in STE pressure estimates
        with h5py.File(vwerp_path, 'r') as f:

            # false lumen dp
            ste_estimations += f['f1_dp'][:].flatten().tolist()
            ste_estimations += f['f2_dp'][:].flatten().tolist()
            ste_estimations += f['f3_dp'][:].flatten().tolist()
            ste_estimations += f['f4_dp'][:].flatten().tolist()
            ste_estimations += f['f5_dp'][:].flatten().tolist()

            # true lumen dp
            ste_estimations += f['t1_dp'][:].flatten().tolist()
            ste_estimations += f['t2_dp'][:].flatten().tolist()
            ste_estimations += f['t3_dp'][:].flatten().tolist()
            ste_estimations += f['t4_dp'][:].flatten().tolist()
            ste_estimations += f['t5_dp'][:].flatten().tolist()
            ste_estimations += f['o_dp'][:].flatten().tolist()

    # convert measurements to numpy array for faster regression and plotting
    cath_measurements = np.asarray(cath_measurements)
    ste_estimations = np.asarray(ste_estimations)

    return cath_measurements, ste_estimations


def hist_plot(cath_measurements, estimations, method, plot_type='hist2d', exclude=False):

    if exclude:
        line_color = 'black' #NOTE I screwed around with this stuff earlier so idk my plotting software has kinda gone haywire but at least I'm producing something cool...
        text_color = 'black'
        min_count = 1
    else:
        line_color = 'white'
        text_color = 'white'
        min_count = 0

    # Regression Stuff
    reg = stats.linregress(cath_measurements, estimations)
    #x = np.array([np.min(cath_measurements), np.max(cath_measurements)])
    x = np.array([-6.0, 20.0])

    if reg.intercept < 0.0:
        reg_stats = f'$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$'
    else:
        reg_stats = f'$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$'

    # Plotting
    fig, ax = plt.subplots(figsize=(10,8))
    ax.text(0.05, 0.95, reg_stats, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=20, color=text_color)
    ax.plot(x, reg.intercept + reg.slope*x, color=line_color, linewidth='3')
    ax.plot(x, x, color=line_color, linestyle='--', linewidth=3)

    # Plotting parameters that depend on plot type
    if plot_type == 'hexbin':
        density = ax.hexbin(cath_measurements, estimations, gridsize=50, extent=x.tolist() + x.tolist(), mincnt=min_count, vmax=10, cmap='inferno')
        cbar = fig.colorbar(density, ax=ax)

    elif plot_type == 'hist2d':
        density = ax.hist2d(cath_measurements, estimations, bins=26, range=[x, x], cmin=min_count, vmax=25, cmap='inferno')
        cbar = fig.colorbar(density[3], ax=ax)

    elif plot_type == 'hist2d_normalized':

        # math shit
        H, xedges, yedges = np.histogram2d(cath_measurements, estimations, bins=26, range=[x, x])   # at some point, will probably want to double-check to make sure I'm doing this stuff correctly

        # find the normalization factor for each column (axis=1 bc H is transposed)
        H_norm = np.maximum(1, np.sum(H, axis=1))
        H = H/H_norm[:,None]

        masked_H = np.ma.array(H, mask=(H == 0))

        cmap = cm.inferno_r
        cmap.set_bad('white', -1e-7) # FIXME: I want 0 to show up on colorbar??? hmmmmm

        # plot
        density = ax.imshow(masked_H.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmax=1.0, cmap=cmap)
        cbar = fig.colorbar(density, ax=ax, format=mtick.PercentFormatter(1.0))
        cbar.set_label(label='Distribution per Column', fontsize=18)


    else:
        print("Undefined plot type!")
        return

    # Plot formatting
    ax.set_aspect('equal')
    ax.set_ylim(bottom=x[0], top=x[1])
    ax.set_xlim(left=x[0], right=x[1])
    ax.set_title(f'{method} Flow Phantom Correlation Plot', fontsize=22)
    ax.set_xlabel(r'$\Delta \mathregular{P_{cath}}$ [mmHg]', fontsize=22)
    ax.set_ylabel(fr'$\Delta \mathregular{{P_{{{method}}}}}$ [mmHg]', fontsize=22)  # honestly do not know how this is working but I'm glad it is... raw f-strings wow
    ax.tick_params(axis='both', which='major', labelsize=16)
    cbar.ax.tick_params(labelsize=16)
    #ax.legend()
    #ax.tick_params(axis='both', which='minor', labelsize=12)
    fig.tight_layout()

    return fig


def main():
    # FIXME: the distributions look oddly similar? Might want to double-check my manual data creation process... CATH is definitely being re-used, but estimations should NOT BE... bit odd...
    # NOTE: MIGHT just want to continue doing regular correlation plots for plane-to-plane data, but definitely use this heatmap idea for the fully-spatial correlation plots?
    # ALSO THERE IS AN INCREDIBLE SHITLOAD OF REDUNDANT CODE HERE -> SPLIT DATA LOADS FOR vWERP and STE into two functions and then make 3rd function for plotting, that way it's the same stuff for ste and vwerp and eventually ppe, unsteady bernoulli
    
    data_dir_prefix = '../../../../../Relative Pressure Estimation/vwerp/judith'

    # might make sense to do additional cath_data load function if I start using the same catheter array for both STE and vWERP (i.e. don't double-up on vWERP measurements)
    
    # maybe want to add option for standard scatter plot as well...

    # load data and package into correct arrays
    cath_vwerp, vwerp_estimates = vwerp_load(data_dir_prefix)
    cath_ste, ste_estimates = ste_load(data_dir_prefix)

    # make plots
    vwerp_fig = hist_plot(cath_vwerp, vwerp_estimates, 'vWERP', plot_type='hist2d_normalized', exclude=True) # FIXME: having a lot of trouble getting the "v" to italicize properly...
    ste_fig = hist_plot(cath_ste, ste_estimates, 'STE', plot_type='hist2d_normalized', exclude=True)
    
    # save figures
    vwerp_fig.savefig('vwerp_correlation.eps')
    ste_fig.savefig('ste_correlation.eps')
    vwerp_fig.savefig('vwerp_correlation.svg')
    ste_fig.savefig('ste_correlation.svg')
    vwerp_fig.savefig('vwerp_correlation.pdf')
    ste_fig.savefig('ste_correlation.pdf')
    vwerp_fig.savefig('vwerp_correlation.png', dpi=400)
    ste_fig.savefig('ste_correlation.png', dpi=400)

    plt.show()


if __name__ == "__main__":
    main()