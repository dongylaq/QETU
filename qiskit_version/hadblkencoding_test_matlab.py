""" Test Hadamard block encoding.

    Interface with matlab to run cvx_qsp and qsppack directly to obtain
    the approximate polynomial and phase factors.
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
import os
import su2_qsp_toolbox as su2qsp
import matplotlib.pyplot as plt

# Start from here
# https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
# to install matlab.engine
#
# In matlab
#
#     cd (fullfile(matlabroot,'extern','engines','python'))
#     system('python setup.py install')
#
# Make sure that the right conda environment is chosen before starting
# matlab. For python > 3.7, matlab >= R2021b is needed.
#
# Make sure that the paths for qsppack, chebfun, cvx have been added to
# matlab's path variable
import matlab.engine


if __name__ == "__main__":
    print("===== test TFIM =====")
    # parameters
    n_qubits = 3
    t_tot = 1/(2*n_qubits)
    n_segments = 3
    g_coupling = 1
    pbc = False
    tfim = hbe.transverse_field_Ising_model(n_qubits, g_coupling, pbc=pbc)
    print("n_qubits = {:d},\t t_tot = {:.3f},\t n_segments = {:d},\ng_coupling = {:.3f},\t pbc = {}".format(
        n_qubits, t_tot, n_segments, g_coupling, pbc
    ))
    # exact and Trotterized Hamiltonian evolution
    H = tfim.Hamiltonian
    U = la.expm(-1j*t_tot*H)
    qc = tfim.build_trotter(t_tot, n_segments, trotter_order=2)
    Uqc = retrieve_unitary_matrix(qc)
    print("second order Trotter error {:.5e}".format(la.norm(U - Uqc,2)))

    # build controlled Hamiltonian evolution
    had_blk_e = hbe.HadamardBlockEncoding()
    had_blk_e.build_from_unitary_TFIM(qc)
    U_ctrl_qc = retrieve_unitary_matrix(had_blk_e.qc)
    U_exact_ctrl = la.block_diag(U.conjugate().transpose(), U)
    print("controlled Trotter error {:.5e}".format(la.norm(U_exact_ctrl - U_ctrl_qc,2)))

    # build SHIFTED controlled Hamiltonian evolution
    shift = np.random.randn()
    U = la.expm(-1j*(t_tot*H+shift*np.eye(2**n_qubits)))
    had_blk_e.build_from_unitary_TFIM(qc, shift = shift)
    U_ctrl_qc = retrieve_unitary_matrix(had_blk_e.qc)
    U_exact_ctrl = la.block_diag(U.conjugate().transpose(), U)
    print("shifted controlled Trotter error {:.5e}".format(la.norm(U_exact_ctrl - U_ctrl_qc,2)))

    print("\n===== test VQE type energy measurement =====")
    state_circ = random_circuit(n_qubits+1, 20)
    state = retrieve_state(state_circ)
    state = state[:2**n_qubits]
    succ_prob = la.norm(state,2)**2
    print("succ prob = {:.3f}".format(succ_prob))
    E_exact = np.real_if_close(state.conjugate() @ H @ state) / succ_prob
    E_approx, _ = hbe.measure_energy_VQEType(tfim, state_circ, 1_000_000)
    print("E_exact = \t{:.5f}\nE_approx = \t{:.5f}".format(E_exact, E_approx))

    print("\n===== test QSVT X rot =====")
    n_qubits = 4
    n_segments = 3
    g_coupling = 1
    pbc = False
    dist = 0.1
    tau = 1.0
    tfim = hbe.transverse_field_Ising_model(n_qubits, g_coupling, pbc=pbc)
    print("n_qubits = {:d},\t t_tot = {:.3f},\t n_segments = {:d},\ng_coupling = {:.3f},\t pbc = {},\t dist = {}".format(
        n_qubits, t_tot, n_segments, g_coupling, pbc, dist
    ))
    H = tfim.Hamiltonian
    # shift and scale the spectrum to be in [dist, pi-dist]
    val_H, vec_H = la.eigh(H)
    val_H = val_H.real
    val_H_min = val_H.min()
    val_H_max = val_H.max()
    c1 = (np.pi-2*dist) / (val_H_max - val_H_min)
    c2 = dist - c1 * val_H_min
    t_tot = tau * c1
    shift = tau * c2
    dd = 10
    phi_seq_su2_half = np.random.randn(dd+1)
    phi_seq_su2 = np.zeros(dd*2+1)
    phi_seq_su2[:dd+1] = phi_seq_su2_half
    phi_seq_su2[dd+1:] = phi_seq_su2_half[-2::-1]
    coef = np.real(su2qsp.build_Cheby_coef_from_phi_LS(phi_seq_su2))
    val_H, vec_H = la.eigh(H)
    val_H = val_H.real
    vals = su2qsp.build_Cheby_fun_from_coef(np.cos(val_H*t_tot + shift), coef, 0)
    mat_exact = vec_H @ np.diag(
        vals
    ) @ vec_H.transpose().conjugate()
    print(t_tot*val_H+shift)
    print(vals)

    phi_qc = hbe.convert_su2_to_Xrot(phi_seq_su2)

    qc = tfim.build_trotter(t_tot, n_segments)
    had_blk_e.build_from_unitary_TFIM(qc, shift = shift)

    print("----- build QSVT circuit without alternating (in tfim) -----")
    state_circ = tfim.apply_qsvt_xrot(had_blk_e.qc, phi_seq_su2)
    circ = retrieve_unitary_matrix(state_circ)
    circ = circ[:2**n_qubits, :2**n_qubits]
    circ = mat_exact[0,0] / circ[0,0] * circ
    print("||mat_exact - circ|| = {:.5f}".format(la.norm(mat_exact - circ, 2)))

    print("----- build QSVT circuit with alternating (in hbe) -----")
    state_circ = hbe.apply_qsvt_xrot(had_blk_e.qc, phi_seq_su2)
    circ = retrieve_unitary_matrix(state_circ)
    circ = circ[:2**n_qubits, :2**n_qubits]
    circ = mat_exact[0,0] / circ[0,0] * circ
    print("||mat_exact - circ|| = {:.5f}".format(la.norm(mat_exact - circ, 2)))


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
    dist = 0.1
    n_shots = 1_000_000
    deg = 20
    tfim = hbe.transverse_field_Ising_model(n_qubits, g_coupling, pbc=pbc)
    print("n_qubits = {:d},\t t_tot = {:.3f},\t n_segments = {:d},\ng_coupling = {:.3f},\t pbc = {},\t dist = {}\ndeg = {:d}".format(
        n_qubits, t_tot, n_segments, g_coupling, pbc, dist, deg
    ))
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

    sigma_to_lambda = lambda x: np.arccos(2.0 * x**2 - 1.0)
    lambda_to_sigma = lambda x: np.sqrt((1.0 + np.cos(x)) / 2)

    mu = 0.5 * (val_H_trans[0] + val_H_trans[1])
    gap = val_H_trans[1] - val_H_trans[0]
    E_min = dist * 0.5
    E_max = np.pi - dist
    
    sigma_min = lambda_to_sigma(E_max)
    sigma_mu = lambda_to_sigma(mu)
    sigma_mu_m = lambda_to_sigma(mu+gap/2)
    sigma_mu_p = lambda_to_sigma(mu-gap/2)
    sigma_max = lambda_to_sigma(E_min)
    
    # Compute phase factors using matlab engine
    print("\nstarting matlab engine..")
    eng = matlab.engine.start_matlab()
    
    print("computing phase factors..")
    opts = {'npts': 400, 'epsil': 0.01, 'fscale': 0.9, 'criteria': 1e-6} 
    # note that np.float64 cannot be accepted by matlab
    phi_seq_matlab = eng.cvx_qsp_unwrap(deg, 
            float(sigma_mu), float(sigma_min), float(sigma_mu_m), 
            float(sigma_mu_p), float(sigma_max), 
            opts['npts'], opts['epsil'], opts['fscale'], 
            opts['criteria'])

    phi_seq_su2 = np.array(phi_seq_matlab).reshape(-1)
    
    # build quantum circuits
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




