""" Ground state preparation for spin problem

    Interface with matlab to run cvx_qsp and qsppack directly to obtain
    the approximate polynomial and phase factors.

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
import os
import su2_qsp_toolbox as su2qsp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # The overall formalism for TFIM implements F(cos(tH)) rather than F(cos(tH/2))
    # in the previous construction. To make the phase factors consistent, t is set to 1/2
    tau = 0.5
    # distance the spectrum separates from the edge 0, pi
    dist = 0.1

    print("\n===== test ground state preparation =====")
    n_qubits = 2
    n_segments = 3
    g_coupling = 1
    pbc = False
    n_shots = 1_000_000
    deg = 20
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

    print("n_qubits = {:d},\t t_tot = {:.3f},\t n_segments = {:d},\ng_coupling = {:.3f},\t pbc = {},\t dist = {}\ndeg = {:d}".format(
        n_qubits, t_tot, n_segments, g_coupling, pbc, dist, deg
    ))


    sigma_to_lambda = lambda x: np.arccos(2.0 * x**2 - 1.0)
    lambda_to_sigma = lambda x: np.sqrt((1.0 + np.cos(x)) / 2)

    mu = 0.5 * (val_H_trans[0] + val_H_trans[1])
    gap = (val_H_trans[1] - val_H_trans[0]) 
    E_min = dist * 0.5
    E_max = np.pi - dist
    
    qsp = QSPPhase()
    phi_seq_su2 = qsp.cvx_qsp_heaviside(deg, 
            E_min, mu - gap/2, mu + gap/2, E_max)
    
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

    coef = np.real(su2qsp.build_Cheby_coef_from_phi_LS(phi_seq_su2))
    x = np.linspace(dist/2, np.pi/2-dist/2, 1000)
    y = su2qsp.build_Cheby_fun_from_coef(np.cos(x), coef, 0)
    plt.figure()
    plt.plot(x,y, label="filter")
    plt.plot(t_tot*val_H+shift, np.zeros(*val_H.shape), "*", label="eig val")
    plt.plot(t_tot*val_H_min+shift, 0, "^", label="grd_e (negate t_tot)")
    plt.legend(loc="upper left")
    plt.show()



