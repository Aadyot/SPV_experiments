import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from functools import lru_cache


# from SPV_utils_deprecated import *

############################################### DATA LOADING FUNCTIONS (txt) #####################################################

def get_data_from_folder(foldername):
    # --- Collect all .txt files ---
    files = glob.glob(foldername + "*.txt")
    files = [f for f in files if os.path.basename(f) != "pinned_cells.txt"]
    timestamps = [float(os.path.splitext(os.path.basename(f))[0]) for f in files]

    # --- Sort files by float timestamps ---
    sorted_indices = np.argsort(timestamps)
    timestamps = np.array(timestamps)[sorted_indices]  # sorted timestamps
    files = [files[i] for i in sorted_indices]         # sorted files

    # --- Load each file into an (N, 2) array ---
    return timestamps, np.array([np.loadtxt(f) for f in files])

def get_all_data(foldername = "./100_Ensemble/N=400_pin=0/N=400/coordinate_p0=3.80_v0=0.50_pin=0_set_", get_displacements = True, n_ensembles = None):
    data = {}
    
    # Find all numbered subdirectories
    base_path = foldername.rstrip('/')
    parent_dir = base_path.rsplit('_set_', 1)[0] + '_set_'
    
    # Get all subdirectories and extract set numbers
    subdirs = glob.glob(parent_dir + "*/")
    print(subdirs)
    set_numbers = sorted([int(os.path.basename(d.rstrip('/\\').split('_')[-1])) 
                         for d in subdirs if os.path.isdir(d)])
    if n_ensembles is not None:
        set_numbers = set_numbers[:n_ensembles]
    
    for i in set_numbers:
        folder = rf"{foldername}{i}/"
        print(folder)
        timestamps, arrays = get_data_from_folder(folder)
        if get_displacements:
            displacement = arrays-arrays[0]
        else:
            displacement = arrays
        N = arrays[0].shape[0]
        data[i] = (timestamps, displacement, N)
    
    return data

############################################### DATA LOADING FUNCTIONS (npy) #####################################################

def get_all_data_npy(filename, get_displacements = True, n_ensembles = None):
    data = {}
    
    # Find all numbered subdirectories
    base_path = filename.rstrip('/')
    parent_dir = base_path.rsplit('_p=0_set_', 1)[0] + '_p=0_set_'
    
    # Get all subdirectories and extract set numbers
    files = sorted(glob.glob(filename + "*.npy") + glob.glob(filename + "*.npz"))
    sets = []
    for f in files:
        try:
            idx = int(os.path.splitext(os.path.basename(f))[0].rsplit('_set_', 1)[1])
            sets.append((idx, f))
        except Exception:
            continue
    
    for idx, f in sorted(sets):
        print(f)
        timestamps, arrays = get_data_from_npy(f)
        if get_displacements:
            arrays = arrays - arrays[0]
        N = arrays[0].shape[0]
        data[idx] = (timestamps, arrays, N)
        if n_ensembles is not None and idx > n_ensembles:
            break
    return data

@lru_cache(maxsize=128)
def get_timestamp_npy(path):
    """Return cached timestamps array from time.npy in the same folder as filename."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"time.npy not found: {path}")
    return np.load(path)

def get_data_from_npy(filename):
    arr = np.load(filename)
    timestamps = get_timestamp_npy(os.path.join(os.path.dirname(os.path.abspath(filename)), "time.npy"))
    return timestamps, arr

############################################## Fs CALCULATION FUNCTIONS #####################################################

def Fs(k, disp):
    #k should be (2,) or (n_theta,2)
    if(len(k.shape) == 1):
        k = np.array([k])
    #disp should be (n_points,2) or (n_time, n_points, 2)
    phase = disp@k.T                  #product over exis with length 2. dim is (n_points, n_theta) or (n_time, n_points, n_theta)
    return np.exp(-1j*phase).mean(axis = -2)   #mean over points. output is (n_theta,) or (n_time, n_theta)

def Fs_avg(k_mag, disp, n_theta, get_squares = False):
    # assert get_squares == True
    #disp should be (n_points,2) or (n_time, n_points, 2)
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint = False)
    ks = k_mag * np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    Fs_vals = Fs(ks, disp)
    if get_squares:
        return Fs_vals.mean(axis = -1) , (Fs_vals**2).mean(axis = -1)     #mean over thetas. output is (1,) or (n_time,)
    return Fs_vals.mean(axis = -1)     #mean over thetas. output is (1,) or (n_time,)

def get_all_Fs(data, k, n_theta = 100, get_squares = False):
    # assert get_squares == True;
    print(get_squares)
    data2 = {}
    for i in data.keys():
        timestamps, displacement, N = data[i]
        Fs_all =(Fs_avg(k, displacement, n_theta, get_squares))
        data2[i] = (timestamps, displacement, N, Fs_all)
    return data2

def MSD(displacement):
    return (displacement**2).sum(axis = -1).mean(axis = -1)
############################################# X4 CALCULATION FUNCTIONS #####################################################

def X4_calc(data, Fs_has_squares = False):
    # assert Fs_has_squares == True
    if Fs_has_squares:
        Fs_sum = np.zeros_like(data[1][-1][0])
        Fs_2_sum = np.zeros_like(data[1][-1][1])
    else:
        Fs_sum = np.zeros_like(data[1][-1])
        Fs_2_sum = np.zeros_like(data[1][-1])
    for i in data.keys():
        if Fs_has_squares:
            _, _, N, (Fs, Fs_2) = data[i]
            Fs_sum += Fs
            Fs_2_sum += Fs_2  #its an average of squares over ks
        else:
            _, _, N, Fs = data[i]
            Fs_sum += Fs
            Fs_2_sum += Fs**2 #its square of average over ks
    X4 =((Fs_2_sum/len(data) - (Fs_sum/len(data))**2)) * N
    return X4


############################################# STRUCTURE FACTOR FUNCTIONS #####################################################

def Sf_iso(k, pos, batch_size = None):
    #k should be (2,) or (n_ks,2)
    if(len(k.shape) == 1):
        k = np.array([k])
    if(len(pos.shape) == 1):
        pos = np.array([pos])
    #pos should be (n_time, n_points, 2)
    N = pos.shape[1]
    if batch_size == None:
        phase = pos@k.T                   #product over exis with length 2. dim is  (n_time, n_points, n_ks)
        rho_k = np.exp(-1j * phase)
        return (np.abs(rho_k.sum(axis = -2))**2) / N    #mean over points. output is  (n_time, n_ks)
    
    n_ks = k.shape[0]
    result = np.zeros((pos.shape[0],n_ks), dtype=np.float64)
    for start in range(0, n_ks, batch_size):      #takes too much time
        print("batch", start/batch_size , "of", n_ks/batch_size)
        end = min(start + batch_size, n_ks)
        k_batch = k[start:end, :]  # shape (batch, 2)
        phase = pos@k_batch.T                   #product over exis with length 2. dim is  (n_time, n_points, batch)
        rho_k = np.exp(-1j * phase)
        result[:, start:end] = (np.abs(rho_k.sum(axis = -2))**2)
    return result / N



############################################ POLAR GRID, STRUCTURE FACTOR FUNCTIONS #####################################################

def get_polar_k_grid(n_k = 200, n_theta = 200, k_max = 50):
    k_radii = np.linspace(0, k_max, n_k)
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)

    kr, th = np.meshgrid(k_radii, thetas, indexing='ij')
    kx = kr * np.cos(th)
    ky = kr * np.sin(th)
    
    # shape (n_k*n_theta, 2)
    return np.stack([kx.ravel(), ky.ravel()], axis=1)  , (n_k, n_theta, k_radii, thetas, kx, ky)

def display_polar_k_grid(S_flat, context):
    n_k, n_theta, k_radii, thetas, kx, ky = context
    S_2d = S_flat.reshape(len(k_radii), len(thetas))  # shape (n_k, n_theta)

    plt.figure(figsize=(6,6))
    plt.scatter(kx.ravel(), ky.ravel(), c=np.log(S_2d.ravel()), s=1, cmap='viridis')
    plt.colorbar(label='S(k)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title('Structure Factor S(kx, ky)')
    plt.axis('equal')
    plt.show()
    
    return S_2d

def get_radial_peaks(S_2d, k_radii, threshold = 2000, xlims = None, ylims = None, y_log_scale = True, title = ""):
    S_radial = S_2d.mean(axis = 1)
    peaks, _ = find_peaks(S_radial)
    k_peaks = k_radii[peaks]
    S_peaks = S_radial[peaks]

    plt.figure(figsize=(6,4))
    plt.title(f"Radial Structure Factor S(k) {title}")
    plt.plot(k_radii, (S_radial), label='S_radial')

    real_peaks = []
    # Add labels only for S > threshold
    for k_val, S_val in zip(k_peaks, S_peaks):
        if S_val > threshold:
            if xlims is not None:
                if not (xlims[0] <= k_val <= xlims[1]):
                    continue
            plt.plot(k_val, S_val)
            plt.text(k_val, S_val, f"k={k_val:.3f}, S={int(S_val)}", 
                    color='red', fontsize=9, ha='center', va='bottom')
            real_peaks.append((k_val, S_val))
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    if y_log_scale:
        plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('S(k) radial')
    plt.legend()
    plt.show()
    
    return real_peaks




############################################ OVERLAP FUNCTIONS #####################################################

def Q_t(r_t, r_0, a=0.3):
    #r_t has dimension (time, N, 2)
    #r_0 has dimension (N, 2)
    delta_r = np.linalg.norm(r_t - r_0, axis=-1)
    return (delta_r<a).mean(axis = -1)

def Q_f(q, r_t, r_0, a=0.3):
    #r_t has dimension (time, N, 2)
    #r_0 has dimension (N, 2)
    #q has dimension (n_q, 2)
    assert (r_0 == 0).all() == False, "r_0 should be position not displacement"
    
    delta_r = np.linalg.norm(r_t - r_0, axis=-1)  
    w = delta_r < a #dimension (time, N)
    f = np.exp(1.0j*np.matmul(q, r_0.T))  #dimension (n_q, N)   
        
    #w*q has dimension (time, n_q, N)
    # return (w[:,np.newaxis,:]*f[np.newaxis,:,:]).sum(axis = -1)  #returns dimension (time, n_q)
    return np.einsum('tn,qn->tq', w, f)

def Q_f_avg(q_mag, r_t, r_0, n_theta = 1, a=0.3):
    #r_t should be (n_time, n_points, 2)
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint = False)
    qs = q_mag * np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    return Q_f(qs, r_t, r_0, a)     #mean over qs. output is (n_time, n_theta)

def get_all_Q_f_s(data, q_mag, n_theta = 1):
    data2 = {}
    for i in data.keys():
        timestamps, positions, N = data[i]
        Q_f_all = Q_f_avg(q_mag, positions, positions[0], n_theta)
        data2[i] = (timestamps, positions, N, Q_f_all)
    return data2

############################################# S4 CALCULATION FUNCTIONS #####################################################

def S4(q, disps, n_theta = 10):
    T1 = 0
    T2 = 0 
    for i in disps.keys():
        timestamps, arrs, N = disps[i]
        Q_pos_data = get_all_Q_f_s(disps, q, n_theta)   #choose 1 direction. not many thetas needed
        Q_neg_data = get_all_Q_f_s(disps, -q, n_theta)
        T1 += (Q_pos_data[i][3] * Q_neg_data[i][3]).mean(axis = -1)
        T2 += (Q_pos_data[i][3]**2).mean(axis = -1)
    T1 /= len(disps.keys())
    T2 = T2/(len(disps.keys()))**2
    return (T1 - T2)/disps[1][2]  #normalize by N