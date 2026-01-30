import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from SPV_utils import *


def run_for_N(N):
    print(f"Starting N={N}")

    disps = get_all_data(
        rf"/home/sumit/2025/9_Sept_pin_3.8_0.5_X4_phase_dia_PRE_25/Pinning_Aug_extenstion/Pinning_0_percent/N={N}/coordinate_p0=3.80_v0=0.50_pin=0_set_",
        # rf"../Pin=0/coordinate_p0=3.80_v0=0.25_pin=0_set_",
        False,
    )    

    # ---- Q(t) and T_alpha ----
    Q_t_avg = np.zeros(len(disps[1][0]), dtype=np.complex128)

    for i in disps.keys():
        timestamps, arrs, _ = disps[i]
        Q_t_avg += Q_t(arrs, arrs[0])

    Q_t_avg /= len(disps)
    T_alpha_ind = np.where(np.abs(Q_t_avg) < 1 / np.e)[0][0]
    T_alpha = timestamps[T_alpha_ind]

    # save T_alpha
    with open(f"T_alpha_more_Q{N}.pkl", "wb") as f:
        pickle.dump(
            {
                "N": N,
                "T_alpha": T_alpha,
                "T_alpha_ind": T_alpha_ind,
            },
            f,
        )

    # ---- S4(q) ----
    S4s = {}
    for q in np.arange(0, 10, 0.05):
        S4s[q] = S4(q, disps)

    # save S4
    with open(f"S4_more_Q_{N}.pkl", "wb") as f:
        pickle.dump(S4s, f)

    print(f"Finished N={N}")
    return N

import sys
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python run_S4.py N [N2 N3 ...]")
    # Ns passed from command line
    Ns = [int(arg) for arg in sys.argv[1:]]

    nproc = min(len(Ns), cpu_count())
    # with Pool(processes=nproc) as pool:
        # pool.map(run_for_N, Ns)
    for N in Ns:
        run_for_N(N)
