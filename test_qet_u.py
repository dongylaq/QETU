""" 
    Test Hadamard block encoding.
"""
from circuit_to_matrix import retrieve_unitary_matrix, retrieve_state
import qet_u as qetu
import numpy as np
import scipy.linalg as la
import su2_qsp_toolbox as su2qsp
from qsp_phase_matlab import QSPPhase
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("===== test TFIM =====")
    # parameters
    n_qubits = 4
    n_segments = 3
    g_coupling = 1
    pbc = False
    # distance the spectrum separates from the edge 0, pi
    dist = 0.1
    n_shots = 1_000_000
    deg = 40
    # The overall formalism for TFIM implements F(cos(tH)) rather than F(cos(tH/2))
    # in the previous construction. To make the phase factors consistent, t is set to 1/2
    tau = 0.5
    tfim = qetu.transverse_field_Ising_model(
        n_qubits    = n_qubits, 
        g_coupling  = g_coupling,
        pbc         = pbc
    )
    # exact and Trotterized Hamiltonian evolution
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

    print(
        f"n_qubits = {n_qubits},\t t_tot = {t_tot},\t n_segments = {n_segments},\n" + \
        f"g_coupling = {g_coupling},\t pbc = {pbc},\t dist = {dist}\ndeg = {deg}"
    )

    U = la.expm(-1j*t_tot*H)
    qc = tfim.build_trotter(
        t_tot           = t_tot, 
        n_segments      = n_segments, 
        trotter_order   = 2
    )
    Uqc = retrieve_unitary_matrix(qc)
    print("second order Trotter error {:.5e}".format(la.norm(U - Uqc,2)))

    # build controlled Hamiltonian evolution
    had_blk_e = qetu.HadamardBlockEncoding()
    had_blk_e.build_from_unitary_TFIM(qc)
    U_ctrl_qc = retrieve_unitary_matrix(had_blk_e.qc)
    U_exact_ctrl = la.block_diag(U.conjugate().transpose(), U)
    print("controlled Trotter error {:.5e}".format(la.norm(U_exact_ctrl - U_ctrl_qc,2)))

    print("\n===== test ground state preparation =====")
    mu = 0.5 * (val_H_trans[0] + val_H_trans[1])
    gap = (val_H_trans[1] - val_H_trans[0])
    # multiply 0.5 for numerical stability, otherwise E_min = dist (not an interval)
    E_min = dist * 0.5
    E_max = np.pi - dist

    qsp = QSPPhase()
    phi_seq_su2 = qsp.cvx_qsp_heaviside(
        deg         = deg, 
        E_min       = E_min,
        E_mu_m      = mu - gap/2, 
        E_mu_p      = mu + gap/2, 
        E_max       = E_max
    )
    
    # build quantum circuits
    qc = tfim.build_trotter(t_tot, n_segments)
    had_blk_e.build_from_unitary_TFIM(
        qc          = qc, 
        shift       = shift
    )

    state_circ = tfim.apply_qsvt_xrot(
        be_qc       = had_blk_e.qc, 
        phi_seq_su2 = phi_seq_su2
    )
    state = retrieve_state(state_circ)
    state = state[:2**n_qubits]
    succ_prob = la.norm(state,2)**2
    print("succ prob = {:.3f}".format(succ_prob))
    state = grd_state[0] / state[0] * state
    print("||grd_state - state_circ|| = {:.5f}".format(la.norm(grd_state - state, 2)))

    E_approx, _ = qetu.measure_energy_VQEType(
        tfim        = tfim, 
        state_circ  = state_circ, 
        meas_shots  = n_shots
    )
    print("E_gs\t\t\t = {:.5f}\nE_recon (proj)\t\t = {:.5f}".format(
        val_H_min, E_approx
    ))

    coef = np.real(su2qsp.build_Cheby_coef_from_phi_LS(phi_seq_su2))
    x = np.linspace(dist/2, np.pi/2-dist/2, 1000)
    y = su2qsp.build_Cheby_fun_from_coef(
        x           = np.cos(x), 
        coef        = coef, 
        parity      = 0
    ) / qsp.opts["fscale"]
    plt.figure()
    plt.plot(x,y, label="filter")
    plt.plot(t_tot*val_H+shift, np.zeros(*val_H.shape), "*", label="eig val")
    plt.plot(t_tot*val_H_min+shift, 0, "^", label="grd_e")
    plt.legend(loc="upper right")
    plt.show()
