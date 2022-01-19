""" Ground state preparation for spin problem on IBMQ.

    Last revision: 01/18/2022
"""
import hadamard_block_encoding as hbe
import numpy as np
from ground_spin_ibmq import get_deg_list, get_backend, PHI_DATA_DIR
import scipy.linalg as la
import os, sys
import pickle
import matplotlib.pyplot as plt

def estimate_alpha(counts):
    keys = list(counts.keys())
    n_qubits = len(keys[0])
    if "0"*n_qubits in keys:
        p0 = counts["0"*n_qubits]
        keys.remove("0"*n_qubits)
    else:
        p0 = 0
    pn0 = [counts[ii] for ii in keys]
    alpha = (p0-np.mean(pn0)) / (p0+np.sum(pn0))
    return alpha

if __name__ == "__main__":
    backend_name = sys.argv[1]
    n_qubits = int(sys.argv[2])
    savepath = os.path.dirname(os.path.abspath(__file__)) + \
        f"/data/ground/ibmq/{backend_name}"
    os.makedirs(savepath, exist_ok=True)
    deg_list = get_deg_list(n_qubits, even_only=True)
    data = []
    for opt_level in [0, 3]:
        savedatapath = savepath + f"/{opt_level}/n{n_qubits}.pkl"
        data.append(pickle.load(open(savedatapath, "rb")))
    *E_approx, val_H_min = np.load(PHI_DATA_DIR+f"/n{n_qubits}/approx.npy")
    n_data = len(data)
    print(f"data is retrieved, # = {n_data}..", flush=True)
    E_approx = E_approx[1::2]
    g_coupling = 4.0
    pbc = False
    tfim = hbe.transverse_field_Ising_model(n_qubits, g_coupling, pbc=pbc)
    E_est = np.zeros((len(deg_list), 2))
    E_est_cor = np.zeros((len(deg_list), 2))
    for ii, data_ii in enumerate(data):
        for jj, _data in enumerate(data_ii):
            alpha = estimate_alpha(_data[0])
            energy_zz = hbe.counts_to_energy(_data[1], "zz")
            energy_x = hbe.counts_to_energy(_data[2], "x")
            energy = - energy_zz - g_coupling * energy_x
            E_est[jj,ii] = energy
            E_est_cor[jj,ii] = energy / alpha
    
    pickle.dump(
        [deg_list, E_approx, E_est, E_est_cor, val_H_min],
        open(savepath + f"/n{n_qubits}_data.pkl", "wb")
    )

    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    lines = ["-","--",":","-."]
    res_lw = 1.5
    markers = ["o","v","^","<",">","s","D","1","2","3","4","8","p","*"]
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    plt.subplots_adjust(top=0.9, bottom=0.225, hspace=0.25, wspace=0.28)
    label_list = [
        "opt level = 0", # "opt level = 0 (corrected)",
        "opt level = 3", # "opt level = 3 (corrected)",
        "noiseless", "exact"
    ]
    l_list = []
    for ii in range(2):
        l_temp, = ax.plot(deg_list, E_est[:,ii], marker = markers[ii], ls = lines[ii], alpha = 1)
        l_list.append(l_temp)
        # l_temp, = ax.plot(deg_list, E_est_cor[:,ii], marker = markers[ii], ls = lines[ii], alpha = 0.25)
        # l_list.append(l_temp)
    l_temp, = ax.plot(deg_list, E_approx, marker = markers[2], ls = lines[2], alpha = 1)
    l_list.append(l_temp)
    l_temp, = ax.plot(deg_list, val_H_min * np.ones_like(deg_list), ls = lines[3], alpha = 1)
    l_list.append(l_temp)
    ax.set_xlabel("degree")
    ax.set_ylabel("energy")
    ax.set_title(f"n_sys_qubits = {n_qubits}")
    fig.legend(
        handles = l_list, 
        labels = label_list, 
        loc = 'lower center', 
        borderaxespad=0.3, 
        ncol = len(l_list)//2, 
        prop={'size': 11}
    )
    fig.savefig(
        savepath + f"/n{n_qubits}.pdf",
        format="pdf", 
        dpi=1000, 
        bbox_inches="tight", 
        pad_inches = 0
    )
