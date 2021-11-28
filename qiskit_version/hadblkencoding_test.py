from racbem import retrieve_unitary_matrix, retrieve_state
from hadamard_block_encoding import HadamardBlockEncoding
from hadamard_block_encoding import build_trotter_TFIM, build_TFIM_Hamiltonian
import numpy as np
import scipy.linalg as la
from qiskit import QuantumCircuit, execute, Aer
from random_circuit import random_circuit
from racbem import QSPCircuit
from qiskit.tools.monitor import job_monitor
import os

PHI_DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../phi_seq"

if __name__ == "__main__":
    print("===== test Hadamard Block-encoding =====")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    print(qc.draw())
    UA = retrieve_unitary_matrix(qc)
    Id = np.eye(UA.shape[0])
    ground_truth = 0.5j * np.block([[Id-UA, Id+UA], [Id+UA, Id-UA]])
    had_blk_e = HadamardBlockEncoding()
    had_blk_e.build_from_unitary(qc)
    UBlkE = retrieve_unitary_matrix(had_blk_e.qc)
    print("||circuit - ground_truth|| = {:.5e}\n".format(la.norm(ground_truth - UBlkE,2)))

    print("random quantum circuit")
    qc = random_circuit(5, 10, basis_gates=["x", "z", "h", "cx"])
    UA = retrieve_unitary_matrix(qc)
    Id = np.eye(UA.shape[0])
    ground_truth = 0.5j * np.block([[Id-UA, Id+UA], [Id+UA, Id-UA]])
    had_blk_e.build_from_unitary(qc)
    UBlkE = retrieve_unitary_matrix(had_blk_e.qc)
    print("||circuit - ground_truth|| = {:.5e}\n".format(la.norm(ground_truth - UBlkE,2)))

    print("===== test TFIM =====")
    n_qubits = 5
    t_tot = 1/(2*n_qubits)
    n_segments = 100
    g_coupling = 1
    pbc = False
    print("n_qubits = {:d},\t t_tot = {:.3f},\t n_segments = {:d},\ng_coupling = {:.3f},\t pbc = {}".format(
        n_qubits, t_tot, n_segments, g_coupling, pbc
    ))
    H = build_TFIM_Hamiltonian(n_qubits, g_coupling, pbc=pbc)
    U = la.expm(-1j*t_tot*H)
    qc1 = build_trotter_TFIM(n_qubits, t_tot, n_segments, g_coupling, trotter_order=1, pbc=pbc)
    Uqc1 = retrieve_unitary_matrix(qc1)
    qc2 = build_trotter_TFIM(n_qubits, t_tot, n_segments, g_coupling, trotter_order=2, pbc=pbc)
    Uqc2 = retrieve_unitary_matrix(qc2)
    print("first order Trotter error {:.5e}".format(la.norm(U - Uqc1,2)))
    print("second order Trotter error {:.5e}".format(la.norm(U - Uqc2,2)))

    # distance the spectrum separates from the edge 0, pi
    dist = 0.1
    # shift and scale the spectrum to be in [dist, pi-dist]
    val_H, vec_H = la.eig(H)
    val_H = val_H.real
    val_H_min = val_H.min()
    val_H_max = val_H.max()
    c1 = (np.pi-2*dist) / (val_H_max - val_H_min)
    c2 = dist - c1 * val_H_min
    t_tot = 1.0 * c1
    angle = 1.0 * c2

    print("\n===== test shifted Hamiltonian =====")
    print("n_qubits = {:d},\t t_tot = {:.3f},\t n_segments = {:d},\ng_coupling = {:.3f},\t pbc = {},\t dist = {}".format(
        n_qubits, t_tot, n_segments, g_coupling, pbc, dist
    ))
    val_H_lt = c1 * val_H + c2
    U_Hlt = vec_H @ np.diag(np.exp(-1j*val_H_lt)) @ vec_H.conjugate().transpose()
    Id = np.eye(*U_Hlt.shape)
    ground_truth = 0.5 * np.block([[Id-U_Hlt, Id+U_Hlt], [Id+U_Hlt, Id-U_Hlt]])
    global_phase = 1j * np.exp(1j*angle/2)
    qc2 = build_trotter_TFIM(n_qubits, t_tot, n_segments, g_coupling, trotter_order=2, pbc=pbc)
    had_blk_e.build_from_unitary(qc2, angle=angle)
    UBlkE = retrieve_unitary_matrix(had_blk_e.qc)
    print("||circuit (2nd order) - ground_truth|| = {:.5e}\n".format(la.norm(global_phase * ground_truth - UBlkE),2))

    print("\n===== test ground state preparation =====")
    n_qubits = 2
    n_segments = 10
    g_coupling = 1
    pbc = False
    dist = 0.1
    n_shots = 1e6
    deg = 20
    print("n_qubits = {:d},\t t_tot = {:.3f},\t n_segments = {:d},\ng_coupling = {:.3f},\t pbc = {},\t dist = {}\ndeg = {:d}".format(
        n_qubits, t_tot, n_segments, g_coupling, pbc, dist, deg
    ))
    H = build_TFIM_Hamiltonian(n_qubits, g_coupling, pbc=pbc)
    # shift and scale the spectrum to be in [dist, pi-dist]
    val_H, vec_H = la.eig(H)
    val_H = val_H.real
    val_H_min = val_H.min()
    val_H_max = val_H.max()
    c1 = (np.pi-2*dist) / (val_H_max - val_H_min)
    c2 = dist - c1 * val_H_min
    t_tot = 1.0 * c1
    angle = 1.0 * c2
    qsp = QSPCircuit(1,1,n_qubits)
    # load phase factors and the scaling constants
    *phi_qc, scale_denom = np.loadtxt(f"{PHI_DATA_DIR}/ground_state/phi_gsp_n{n_qubits}_d{deg}.txt")
    *phi_qc_H, scale_numer = np.loadtxt(f"{PHI_DATA_DIR}/ground_state/phi_gse_n{n_qubits}_d{deg}.txt")
    # build quantum circuits
    backend = Aer.get_backend('qasm_simulator')
    qc2 = build_trotter_TFIM(n_qubits, t_tot, n_segments, g_coupling, trotter_order=2, pbc=pbc)
    had_blk_e.build_from_unitary(qc2, angle=angle)
    qsp.build_circuit(had_blk_e.qc, phi_qc, realpart=True, measure=True)
    compiled_circ = qsp.qc
    job = execute(compiled_circ, backend, shots = n_shots, optimization_level = 0)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts(compiled_circ)
    if "00" in counts.keys():
        prob_meas_denom = float(counts["00"]) / n_shots
    else:
        prob_meas_denom = 0.0
    qsp.build_circuit(had_blk_e.qc, phi_qc_H, realpart=True, measure=True)
    compiled_circ = qsp.qc
    job = execute(compiled_circ, backend, shots = n_shots, optimization_level = 0)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts(compiled_circ)
    if "00" in counts.keys():
        prob_meas_numer = float(counts["00"]) / n_shots
    else:
        prob_meas_numer = 0.0
    E_recon2_lt = ((prob_meas_numer/scale_numer) / (prob_meas_denom/scale_denom))
    E_recon2 = (E_recon2_lt - c2) / c1

    qsp.build_circuit(had_blk_e.qc, phi_qc, realpart=True, measure=False)
    compiled_circ = qsp.qc
    state = retrieve_state(compiled_circ)
    grd_state = state[:2**n_qubits]
    grd_state = grd_state / la.norm(grd_state, 2)
    E_recon = np.real(grd_state @ H @ grd_state)
    print("E_gs\t\t\t = {:.5f}\nE_recon (proj)\t\t = {:.5f}\nE_recon (2 meas)\t = {:.5f}".format(
        val_H_min, E_recon, E_recon2
    ))




