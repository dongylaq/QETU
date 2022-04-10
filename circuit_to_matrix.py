"""
    Helper functions convert quantum circuit (qiskit) to np.ndarray
    
    Authors:
        Yulong Dong     dongyl (at) berkeley (dot) edu
    Version: 1.0
    Last revision: 04/2022
"""
import numpy as np
from qiskit import Aer, execute
from qiskit.quantum_info import Operator as get_unitary
from qutip import Qobj
from typing import Any
from nptyping import NDArray

def retrieve_unitary_matrix(
        qcircuit: "QuantumCircuit"
    ) -> NDArray[(Any, Any), complex]:
    UA = get_unitary(qcircuit).data
    n_tot_qubit = int(np.log2(UA.shape[0]))
    # qiskit's column-major order
    UA_Qobj = Qobj(UA, dims=[[2]*n_tot_qubit, [2]*n_tot_qubit])
    # python's row-major order
    UA_Qobj_rowmajor = UA_Qobj.permute(np.arange(0,n_tot_qubit)[::-1])
    UA = np.array(UA_Qobj_rowmajor)
    return UA

def retrieve_state(
        qcircuit: "QuantumCircuit"
    ) -> NDArray[(Any,), complex]:
    backend = Aer.get_backend("statevector_simulator")
    n_tot_qubit = qcircuit.num_qubits
    job = execute(qcircuit, backend)
    result = job.result()
    state = result.get_statevector(qcircuit)
    # qiskit's column-major order
    state_Qobj = Qobj(np.array(state), dims=[[2]*n_tot_qubit, [1]])
    # python's row-major order
    state_Qobj_rowmajor = state_Qobj.permute(np.arange(0,n_tot_qubit)[::-1])
    state_rowmajor = np.array(state_Qobj_rowmajor)[:,0]
    return state_rowmajor