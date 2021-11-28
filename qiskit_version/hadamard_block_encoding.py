"""Module for Hadamard Block-encoding implemented using IBM's Qiskit.

Authors:
    Yulong Dong     dongyl (at) berkeley (dot) edu
Version: 1.0
Last revision: 11/2021
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

class HadamardBlockEncoding(object):
    """Implement the Hadamard block-encoding of a unitary"""
    __n_be_qubit = 1

    def __init__(self):
        pass

    def build_from_unitary(self, qc: QuantumCircuit, angle: float = 0) -> None:
        """Build a Hadamard block-encoding circuit from a given unitary (circuit)
            -[H]--[RZ(-(pi+angle)]-.-[H]-
                                   |
            ----------------------[U]----
            unitary rep: global phase (1j * exp(1j*angle/2)) *
            1/ [I-V, I+V]
            /2 [I+V, I-V] where V = exp(-1j*angle)*U
        """
        self.n_sys_qubit = qc.num_qubits
        self.n_tot_qubit = self.__n_be_qubit + self.n_sys_qubit
        self.qr = QuantumRegister(self.n_tot_qubit, 'q')
        if hasattr(self, 'qc'):
            self.qc.data.clear()
        self.qc = QuantumCircuit(self.qr)
        self.qc.h(self.qr[0])
        # self.qc.z(self.qr[0])
        self.qc.rz(-(np.pi+angle), self.qr[0])
        # if angle != 0:
        #     self.qc.rz(-angle, self.qr[0])
        self.qc.append(qc.control(1).to_instruction(), self.qr)
        self.qc.h(self.qr[0])

def build_trotter_TFIM(
        n_qubits: int, 
        t_tot: float, 
        n_segments: int, 
        g_coupling: float, 
        trotter_order: int = 2, 
        pbc: bool = False
    ) -> QuantumCircuit:
    # currently only support up to the second order
    assert trotter_order in [1,2]
    qc = QuantumCircuit(n_qubits)

    def _build_trotter_TFIM_append_ZZ(tau_param):
        nonlocal qc
        for ii in range(0,n_qubits-1,2):
            qc.rzz(-2*tau_param, ii, ii+1)
        for ii in range(1,n_qubits-1,2):
            qc.rzz(-2*tau_param, ii, ii+1)
        if pbc:
            qc.rzz(-2*tau_param, n_qubits-1, 0)

    def _build_trotter_TFIM_append_X(tau_param):
        nonlocal qc
        for ii in range(n_qubits):
            qc.rx(-2*tau_param*g_coupling, ii)
    
    tau = t_tot / n_segments
    _build_trotter_TFIM_append_X(tau if (trotter_order == 1) else tau/2)
    for _ in range(n_segments-1):
        _build_trotter_TFIM_append_ZZ(tau)
        _build_trotter_TFIM_append_X(tau)
    _build_trotter_TFIM_append_ZZ(tau)
    if trotter_order == 2:
        _build_trotter_TFIM_append_X(tau/2)
    return qc
    
def build_TFIM_Hamiltonian(
        n_qubits: int, 
        g_coupling: float,
        pbc: bool = False
    ) -> np.ndarray:
    Zg = np.array([[1,0],[0,-1]])
    ZZ = np.kron(Zg, Zg)
    Xg = np.array([[0,1],[1,0]])
    H = np.zeros((2**n_qubits, 2**n_qubits))
    for ii in range(n_qubits):
        temp = np.kron(np.kron(np.eye(2**ii), Xg), np.eye(2**(n_qubits-1-ii)))
        H = H - g_coupling*temp
        if ii==n_qubits-1:
            if pbc:
                temp = np.kron(Zg, np.eye(2**(n_qubits-2)))
                temp = np.kron(temp, Zg)
            else:
                temp = 0
        else:
            temp = np.kron(np.kron(np.eye(2**ii), ZZ), np.eye(2**(n_qubits-2-ii)))
        H = H - temp
    return H