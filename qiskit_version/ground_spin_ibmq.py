""" Ground state preparation for spin problem on IBMQ.

    Last revision: 01/18/2022
"""
import hadamard_block_encoding as hbe
import numpy as np
import scipy.linalg as la
from qiskit import execute, Aer, IBMQ
from qiskit.tools.monitor import job_monitor
import os, sys
import pickle

def get_deg_list(n_qubits, even_only = True):
    if n_qubits in [2, 3, 4]:
        end = 20
    elif n_qubits in [5, 6]:
        end = 30
    elif n_qubits == 7:
        end = 40
    elif n_qubits == 8:
        end = 60
    else:
        raise ValueError("invalid n_qubits")
    return range(int(even_only)+1, end+1, int(even_only)+1)

def get_backend(backend_name=None):
    if backend_name == None:
        backend = Aer.get_backend('qasm_simulator')
    else:
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub="ibm-q-research")
        backend = provider.get_backend(backend_name)
    return backend

PHI_DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../phi_seq/ground_state"

if __name__ == "__main__":
    backend_name = sys.argv[1]
    n_qubits = int(sys.argv[2])
    savepath = os.path.dirname(os.path.abspath(__file__)) + \
        f"/data/ground/ibmq/{backend_name}"
    os.makedirs(savepath, exist_ok=True)
    deg_list = get_deg_list(n_qubits, even_only=True)
    savedatapath = savepath + f"/n{n_qubits}.pkl"
    if os.path.exists(savedatapath):
        data =  pickle.load(open(savedatapath, "rb"))
    else:
        data = []
    n_data = len(data)
    print(f"backup state is retrieved, # = {n_data}..", flush=True)
    deg_list = deg_list[n_data:]
    n_job = len(deg_list)
    backend = get_backend(backend_name=backend_name)
    tot_q_device = backend.configuration().n_qubits
    assert tot_q_device >= n_qubits + 1

    n_segments = 3
    g_coupling = 4.0
    pbc = False
    n_shots = 8192
    tfim = hbe.transverse_field_Ising_model(n_qubits, g_coupling, pbc=pbc)
    if deg_list:
        deg = deg_list[0]
        *phi_seq_su2, c1, c2, _ = np.loadtxt(PHI_DATA_DIR+f"/n{n_qubits}/phi_gsp_n{n_qubits}_d{deg}.txt")
        tau = 0.5
        t_tot = tau * c1
        shift = tau * c2
        # build quantum circuits
        had_blk_e = hbe.HadamardBlockEncoding()
        qc = tfim.build_trotter(t_tot, n_segments)
        had_blk_e.build_from_unitary_TFIM(qc, shift = shift)
    else:
        print("completes writing data..",flush=True)
        exit()

    for ii, deg in enumerate(deg_list):
        job_list = []
        counts = []
        *phi_seq_su2, _, _, _ = np.loadtxt(PHI_DATA_DIR+f"/n{n_qubits}/phi_gsp_n{n_qubits}_d{deg}.txt")
        circ_wo_rot = hbe.apply_qsvt_xrot(had_blk_e.qc, phi_seq_su2, measure_full=True, turn_off_rot=True)
        state_circ = hbe.apply_qsvt_xrot(had_blk_e.qc, phi_seq_su2)
        circ_zz, circ_x = hbe.circuit_energy_VQEType(state_circ)
        for compiled_circ in [circ_wo_rot, circ_zz, circ_x]:
            job = execute(compiled_circ, backend, shots = n_shots, optimization_level = 3)
            job_list.append([job, compiled_circ])
        for job, compiled_circ in job_list:
            result = job.result()
            counts.append(result.get_counts(compiled_circ))
        data.append(counts)
        pickle.dump(data, open(savedatapath, "wb"))
        print(f"complete writing data {ii+1}/{n_job}, d = {deg}..",flush=True)