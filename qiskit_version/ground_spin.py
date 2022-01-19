""" Ground state preparation for spin problem

    Interface with matlab to run cvx_qsp and qsppack directly to obtain
    the approximate polynomial and phase factors.

    Measure the energy with respect to polynomial degree.

    Last revision: 01/09/2022
"""
from racbem import retrieve_unitary_matrix, retrieve_state
import hadamard_block_encoding as hbe
import numpy as np
import scipy.linalg as la
from qiskit import QuantumCircuit, execute, Aer
from random_circuit import random_circuit
from racbem import QSPCircuit
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer import AerSimulator
from qsp_phase_matlab import QSPPhase
import os, sys
import su2_qsp_toolbox as su2qsp
import matplotlib.pyplot as plt
import json

PHI_DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../phi_seq"

if __name__ == "__main__":
    # The overall formalism for TFIM implements F(cos(tH)) rather than F(cos(tH/2))
    # in the previous construction. To make the phase factors consistent, t is set to 1/2
    tau = 0.5
    # distance the spectrum separates from the edge 0, pi
    dist = 0.1

    print("\n===== test ground state preparation =====")
    n_qubits = int(sys.argv[1])
    n_segments = 3
    g_coupling = 4.0
    pbc = False
    n_shots = 1_000_000
    tfim = hbe.transverse_field_Ising_model(n_qubits, g_coupling, pbc=pbc)
    H = tfim.Hamiltonian
    # shift and scale the spectrum to be in [dist, pi-dist]
    val_H, vec_H = la.eigh(H)
    val_H = val_H.real
    val_H_min = val_H.min()
    val_H_max = val_H.max()
    grd_state = vec_H[:, val_H.argmin()]

    c1 = (np.pi-2*dist) / (val_H_max - val_H_min)
    c2 = dist - c1 * val_H_min
   
    val_H_trans = val_H * c1 + c2
    
    t_tot = tau * c1
    shift = tau * c2

    if not os.path.exists(PHI_DATA_DIR+f"/ground_state/n{n_qubits}"):
        os.mkdir(PHI_DATA_DIR+f"/ground_state/n{n_qubits}")

    print("n_qubits = {:d},\t t_tot = {:.3f},\t n_segments = {:d},\ng_coupling = {:.3f},\t pbc = {},\t dist = {}\n".format(
        n_qubits, t_tot, n_segments, g_coupling, pbc, dist))

    mu = 0.5 * (val_H_trans[0] + val_H_trans[1])
    gap = (val_H_trans[1] - val_H_trans[0]) 
    E_min = dist * 0.5
    E_max = np.pi - dist

    print("Before transformation")
    print(f"H_min = {val_H_min}, H_max = {val_H_max}\n")
    print("After transformation")
    print(f"H_min = {val_H_trans[0]}, H_max = {val_H_trans[-1]}")
    print(f"mu = {mu}, gap = {gap}\n")
    
    qsp = QSPPhase()

    deg_list = np.arange(start=1, stop=61, step=1)
    E_approx_list = np.zeros_like(deg_list, dtype=float)

    for i, deg in enumerate(deg_list):
        print()
        print(f"deg = {deg}")

        phi_seq_su2 = qsp.cvx_qsp_heaviside(deg, 
                E_min, mu - gap/2, mu + gap/2, E_max)
        
        np.savetxt(
            PHI_DATA_DIR+f"/ground_state/n{n_qubits}/phi_gsp_n{n_qubits}_d{deg}.txt",
            np.concatenate((phi_seq_su2, np.array([c1,c2,qsp.opts["fscale"]],dtype=float))),
            fmt = "%.15f"
        )
        
        # build quantum circuits
        had_blk_e = hbe.HadamardBlockEncoding()
        qc = tfim.build_trotter(t_tot, n_segments)
        had_blk_e.build_from_unitary_TFIM(qc, shift = shift)
    
        state_circ = tfim.apply_qsvt_xrot(had_blk_e.qc, phi_seq_su2)
        state = retrieve_state(state_circ)
        state = state[:2**n_qubits]
        succ_prob = la.norm(state,2)**2
        print("succ prob = {:.3f}".format(succ_prob))
        state = grd_state[0] / state[0] * state
        print("||grd_state - state_circ|| = {:.5f}".format(la.norm(grd_state - state, 2)))
    
        E_approx, _ = hbe.measure_energy_VQEType(tfim, state_circ, n_shots)
        print("E_gs\t\t\t = {:.5f}\nE_recon (proj)\t\t = {:.5f}".format(
            val_H_min, E_approx
        ))

        E_approx_list[i] = E_approx
    
    np.save(
        PHI_DATA_DIR+f"/ground_state/n{n_qubits}/approx.npy",
        np.append(E_approx_list, val_H_min)
    )

    E_error_list = np.abs(E_approx_list - val_H_min)
    with open(PHI_DATA_DIR+f"/ground_state/n{n_qubits}/precision.txt", "w") as f:
        data = dict(zip(deg_list.tolist(), E_error_list.tolist()))
        data["exact"] = val_H_min
        f.write(
            json.dumps(data, indent=4, separators=(',', ': '))
        )
    
    # plt.figure()
    # plt.plot(deg_list, E_error_list)
    # plt.xlabel('deg')
    # plt.ylabel('Error')
    # plt.show()

    plt.figure()
    plt.plot(deg_list, E_approx_list, '-o', label='Approx')
    plt.plot(deg_list, val_H_min * np.ones_like(deg_list), '--', 
            label='Exact')
    plt.xlabel('deg')
    plt.ylabel('Energy')
    fig = plt.gcf()
    fig.savefig(PHI_DATA_DIR+f"/ground_state/n{n_qubits}/precision.pdf", 
        format='pdf', dpi=1000, bbox_inches='tight', pad_inches = 0)
