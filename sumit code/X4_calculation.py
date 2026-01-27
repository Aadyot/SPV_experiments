import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob

## Parameters:
total_steps=1e5
dt=0.01
p0=3.80
v0=0.2
pinning=0
N_all=[1024]
N_total=1
plt.figure(figsize=(8,6))   # create one figure for all N
ax = plt.gca()              # main axis

# # Create inset axis (positioned inside main plot)
# show_inset = False   # change to True if you want inset
# if show_inset:
#     ax_inset = inset_axes(ax, width="45%", height="45%", loc="upper right")  
# # loc can be 'upper right', 'upper left', etc.




for N in N_all:
    folder='N=%d'%N

    # Load first file to get length of Fs
    file0 = np.loadtxt('%s/%d_cells_tf=1000.0_p0=3.80_v0=0.50_Dr=1.000_dt=0.010_p=0_1.txt'%(folder, N), skiprows=1)
    time = file0[:, 0]
    Fs0 = file0[:, 2]

    Fs_av= np.zeros_like(Fs0)
    Fs2 = np.zeros_like(Fs0)

    # Loop over all runs
    for i in range(1, N_total+1):
        file1 = np.loadtxt('%s/%d_cells_tf=1000.0_p0=3.80_v0=0.50_Dr=1.000_dt=0.010_p=0_%d.txt' %(folder, N, i), skiprows=1)
        time = file1[:, 0]
        Fs = file1[:, 2]
        Fs_av += Fs
        Fs2 += Fs**2

    Fs_av = Fs_av / N_total
    Fs2 = Fs2 / N_total
    
    # Compute X4(t)
    X4 = N*(Fs2 - Fs_av**2)
    
    # Find max and rescale
    imax = np.argmax(X4)
    t_max = time[imax]
    X4_max = X4[imax]
    time_scaled = time / t_max
    X4_scaled = X4 / X4_max

    # Main plot (rescaled)
    ax.plot(time_scaled, X4_scaled, 'o-', label=f'N={N}')

    # Inset plot (raw X4 vs t)
    if show_inset:
        ax_inset.plot(time, X4, '-', label=f'N={N}')

# Final figure settings
ax.set_xlabel(r'$t/t_{\max}$')
ax.set_ylabel(r'$X_4(t)/X_{4,\max}$')
ax.legend()
ax.grid(True, ls='--', alpha=0.5)

ax.set_xscale("log")
ax.set_title('p0=%0.2f; v0=%0.2f; Ensemble=%d'%(p0, v0, N_total))
'''
# set log-log scale
ax.set_yscale("log")

# set x-axis limits (from 0.03 to auto)
ax.set_xlim(0.01, None)
ax.set_ylim(0.01, None)
'''

#ax.set_title('Rescaled $X_4(t)$ for different system sizes')




# Inset settings
if show_inset:
    ax_inset.set_xscale('log')   # usually X4 vs t is shown on log scale
    ax_inset.set_xlabel(r'$t$', fontsize=9)
    ax_inset.set_ylabel(r'$X_4(t)$', fontsize=9)
    ax_inset.tick_params(axis='both', which='major', labelsize=8)

plt.savefig("N_all_X4_with_inset.png", dpi=300)
plt.show()

