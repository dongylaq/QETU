"""Module for Hadamard Block-encoding implemented using IBM's Qiskit.

Authors:
    Yulong Dong     dongyl (at) berkeley (dot) edu
Version: 1.0
Last revision: 01/2022
"""

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from typing import Any, Tuple, Union, Dict, Optional, Literal
from nptyping import NDArray
from itertools import product as cartesian

class HadamardBlockEncoding(object):
    """Implement the Hadamard block-encoding of a unitary"""
    __n_be_qubit = 1

    def __init__(self):
        pass

    def build_from_unitary(self, qc: "QuantumCircuit", angle: float = 0) -> None:
        """
            Build a Hadamard block-encoding circuit from a given unitary (circuit)
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
        self.qc.rz(-(np.pi+angle), self.qr[0])
        self.qc.append(qc.control(1).to_instruction(), self.qr)
        self.qc.h(self.qr[0])

    def build_from_unitary_TFIM(
            self, 
            qc: "QuantumCircuit",
            shift: Optional[float] = None
        ) -> None:
        """
            Build a Hadamard block-encoding circuit from a given TFIM circuit
            -[exp(i shift Z)]--.-------------.--
                               |             |
            ------------------[A]--[U^dag]--[A]-
            A is a matrix anticommuting with H_1 and H_2
            A = Y \otimes Z \otimes Y \otimes Z ...
            U \approx exp(-1j*t*(H_1+H_2))
            unitary rep: no global phase                [V^dag, 0]
            |0x0| \otimes V^\dagger + |1x1| \otimes V   [0,     V]
            where V = U exp(-i shift) = exp(-i (t H + shift I))
        """
        self.n_sys_qubit = qc.num_qubits
        self.n_tot_qubit = self.__n_be_qubit + self.n_sys_qubit
        self.qr = QuantumRegister(self.n_tot_qubit, 'q')
        if hasattr(self, 'qc'):
            self.qc.data.clear()
        self.qc = QuantumCircuit(self.qr)
        if shift is not None:
            self.qc.rz(-2*shift, 0)
        for ii in range(self.n_sys_qubit):
            if ii % 2 == 0:
                self.qc.cy(*[0,ii+1])
            else:
                self.qc.cz(*[0,ii+1])
        self.qc.append(qc.inverse().to_instruction(), self.qr[1:])
        for ii in range(self.n_sys_qubit):
            if ii % 2 == 0:
                self.qc.cy(*[0,ii+1])
            else:
                self.qc.cz(*[0,ii+1])

class transverse_field_Ising_model(object):
    """
        Transverse field Ising model.
        H = - \sum_i Z_iZ_{i+1} - g \sum_i X_i
        Attributes:
            n_qubits:   the number of qubits of interaction
            g_coupling: coupling constant g in the Hamiltonian
            pbc:        whether the periodic boundary condition is imposed
    """
    def __init__(
        self,
        n_qubits: int,
        g_coupling: float,
        pbc: bool = False
    ) -> None:
        self.n_qubits = n_qubits
        self.g_coupling = g_coupling
        self.pbc = pbc

    @property
    def parameters(self) -> Tuple[int, float, bool]:
        return self.n_qubits, self.g_coupling, self.pbc

    @property
    def Hamiltonian(self) -> np.ndarray:
        """
            Compute the exact TFIM Hamiltonian
                H = - \sum_i Z_i Z_(i+1) - g \sum_i X_i
                pbc:    the periodic boundary condition is imposed or not
                        True: include Z_nZ_0, False: exclude Z_nZ_0
        """
        n_qubits, g_coupling, pbc = self.parameters
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

def build_trotter_TFIM(
        tfim: transverse_field_Ising_model,
        t_tot: float, 
        n_segments: int, 
        trotter_order: int = 2, 
    ) -> QuantumCircuit:
    """
        Build the quantum circuit of the Trotterized evolution of TFIM.
        Unitary Rep: exp(-i t H) where H = -\sum Z_iZ_{i+1} - g \sum X_i
        Each Trotter step:
        -[RZZ(1,2)]------------[RX(1)]-
        -[RZZ(1,2)]-[RZZ(2,3)]-[RX(2)]-
        -[RZZ(3,4)]-[RZZ(2,3)]-[RX(3)]-
        -[RZZ(3,4)]------------[RX(4)]-
    """
    # currently only support up to the second order
    assert trotter_order in [1,2]
    n_qubits, g_coupling, pbc = tfim.parameters
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

def apply_qsvt_xrot(
    be_qc: "QuantumCircuit",
    phi_seq: NDArray[(Any,), float],
    measure: bool = False,
    measure_full: bool = False,
    init_state: Optional[Union["QuantumCircuit", str]] = None
) -> "QuantumCircuit":
    """
        Build a QSVT circuit using the phase factors in the Rx convention.
                            (be_qc)                (be_qc_dag)      meas,   meas_full
        |0>   -[RX(phi_0)]-|  ctrl  |-[RX(phi_1)]-|  ctrl  |- ...   #       #
        |psi> -------------|U+/0,U/1|-------------|U/0,U+/1|- ...   _       #
        Args:
            be_qc:      ctrl-(U+/0,U/1), yielded from HadamardBlockEncoding
            phi_seq:    (phi_0,phi_1,...,phi_1,phi_0) symmetric phase factors
            init_state: circuit preparing |psi>, or bit-str rep in comp basis
    """
    # only one meaurement keyword can be specified as True
    assert (measure and measure_full) is not True
    n_tot_qubit = be_qc.num_qubits
    n_be_qubit = HadamardBlockEncoding._HadamardBlockEncoding__n_be_qubit
    if measure:
        qc = QuantumCircuit(n_tot_qubit, n_be_qubit)
    elif measure_full:
        qc = QuantumCircuit(n_tot_qubit, n_tot_qubit)
    else:
        qc = QuantumCircuit(n_tot_qubit)
    qr = qc.qubits
    # initial state preparation
    if init_state is not None:
        if isinstance(init_state, QuantumCircuit):
            qc.append(init_state.to_instruction(), qr[n_be_qubit:])
        elif isinstance(init_state, str):
            assert len(init_state) == n_tot_qubit-n_be_qubit
            for ii, bit in enumerate(init_state):
                if bit == "1":
                    qc.x(qr[ii+n_be_qubit])
        else:
            raise TypeError(
                    "init_state must be an instance of QuantumCircuit ot str."
                )
    # build alternate X-phase modulation sequence
    dag = False
    be_qc_dag = be_qc.inverse()
    qc.rx(-2*phi_seq[-1], qr[0])
    for phi in reversed(phi_seq[:-1]):
        if dag:
            qc.append(be_qc_dag.to_instruction(), qr)
        else:
            qc.append(be_qc.to_instruction(), qr)
        dag = not dag
        qc.rx(-2*phi, qr[0])
    # add measurements
    if measure or measure_full:
        cr = qc.clbits
        qc.measure(qr[:len(cr)], cr)
    return qc

def convert_su2_to_Xrot(
    phi_seq_su2: NDArray[(Any,), float],
    half: bool = False
) -> NDArray[(Any,), float]:
    """
        Convert the phase factors from SU2 version to that of X rotations.
        Given a set of phase factors Phi of degree d w/ Re[<0|.|0>] = F(x)
            if d/2 is even: Phi parameterizes (-1)^(d/2) F(x) in real comp.
            if d/2 is odd: (phi0-pi/2, phi1, ..., phid-pi/2)->(-1)^(d/2) F(x).
        Convert it to usual Z-rotation convention gives X-rotation convention
            if d/2 is even:
                (phi0+pi/4, phi1+pi/2, phi2+pi/2, ..., phu(d-1)+pi/2, phid+pi/4)
            if d/2 is odd:
                (phi0-pi/4, phi1+pi/2, phi2+pi/2, ..., phu(d-1)+pi/2, phid-pi/4)
        Attributes:
            phi_seq_su2: SU2 phase factors of even degree
            half:        only (phi0,...phi(d/2)) (left half) is given
    """
    if half:
        deg = (len(phi_seq_su2)-1) * 2
        phi_seq = np.zeros(deg+1)
        phi_seq[1:deg//2+1] = phi_seq_su2[1:] + np.pi/2
        phi_seq[deg//2+1:-1] = phi_seq_su2[-2:0:-1] + np.pi/2
        phi_seq[0] = phi_seq_su2[0]
        phi_seq[-1] = phi_seq_su2[0]
    else:
        deg = len(phi_seq_su2)-1
        assert deg % 2 == 0
        phi_seq = np.zeros(deg+1)
        phi_seq[1:-1] = phi_seq_su2[1:-1] + np.pi/2
        phi_seq[0] = phi_seq_su2[0]
        phi_seq[-1] = phi_seq_su2[-1]
    if (deg//2) % 2 == 0:
        phi_seq[0] += np.pi/4
        phi_seq[-1] += np.pi/4
    else:
        phi_seq[0] -= np.pi/4
        phi_seq[-1] -= np.pi/4
    return phi_seq

def convert_Zrot_to_Xrot(
    phi_seq_zrot: NDArray[(Any,), float]
) -> NDArray[(Any,), float]:
    """
        Convert the phase factors from Z rotations to that of X rotations.
            if d/2 is even:
                xphi = zphi
            if d/2 is odd:
                xphi0 = zphi0 - pi/2, xphi(d) = zphi(d) - pi/2
                xphi(i) = zphi(i) i = 1, ..., d-1
    """
    deg = len(phi_seq_zrot)-1
    assert deg % 2 == 0
    phi_seq = phi_seq_zrot
    if (deg//2) % 2 == 1:
        phi_seq[0] -= np.pi/2
        phi_seq[-1] -= np.pi/2
    return phi_seq

def measure_energy_VQEType(
    tfim: transverse_field_Ising_model,
    state_circ: "QuantumCircuit",
    meas_shots: Union[int, Tuple[int, int]],
    noise_model: Optional[Union["NoiseModel", Tuple["NoiseModel", "NoiseModel"]]] = None
) -> Tuple[float, Tuple[Dict[str, int], Dict[str, int]]]:
    """
        Build quantum circuits for the energy measurement.
        Input
            |0>   -[state_circ]- # w/ 0
            |0^n> -[state_circ]- |psi>
        Circuits
        (1) measuring in comp. basis
            |0>   -[state_circ]- # w/ 0
            |0^n> -[state_circ]- # -> dict
        (2) measuring in Had. basis
            |0>   -[state_circ]-------- # w/ 0
            |0^n> -[state_circ]-[H(n)]- # -> dict
    """
    if isinstance(meas_shots, int):
        meas_shots_comp = meas_shots
        meas_shots_had = meas_shots
    elif isinstance(meas_shots, tuple):
        assert len(meas_shots) == 2
        meas_shots_comp, meas_shots_had, = meas_shots
    else:
        raise TypeError("meas_shots must be int or Tuple[int]")
    if (noise_model is None) or isinstance(noise_model, NoiseModel):
        noise_model_comp = noise_model
        noise_model_had = noise_model
    elif isinstance(noise_model, tuple):
        assert len(noise_model) == 2
        noise_model_comp, noise_model_had, = noise_model
        assert isinstance(noise_model_comp, NoiseModel) and\
            isinstance(noise_model_had, NoiseModel)
    else:
        raise TypeError("noise_model must be None, NoiseModel or Tuple[NoiseModel]")
    _, g_coupling, _ = tfim.parameters
    state_circ_wo_meas = state_circ.remove_final_measurements(inplace = False)
    n_tot_qubits = state_circ_wo_meas.num_qubits
    backend = AerSimulator()
    # Circuit (1)
    circ_comp = QuantumCircuit(n_tot_qubits, n_tot_qubits)
    circ_comp.append(state_circ_wo_meas.to_instruction(), range(n_tot_qubits))
    circ_comp.measure_all(add_bits = False)
    job = execute(circ_comp, backend, shots = meas_shots_comp, optimization_level = 0)
    res_comp = job.result()
    counts_comp = res_comp.get_counts(circ_comp)
    energy_zz = counts_to_energy(counts_comp, "zz")
    # Circuit (2)
    circ_had = QuantumCircuit(n_tot_qubits, n_tot_qubits)
    circ_had.append(state_circ_wo_meas.to_instruction(), range(n_tot_qubits))
    for qri in range(1,n_tot_qubits):
        circ_had.h(qri)
    circ_had.measure_all(add_bits = False)
    job = execute(circ_had, backend, shots = meas_shots_had, optimization_level = 0)
    res_had = job.result()
    counts_had = res_had.get_counts(circ_had)
    energy_x = counts_to_energy(counts_had, "x")
    energy = -energy_zz - g_coupling * energy_x
    return energy, (counts_comp, counts_had)

def counts_to_energy(
    counts: Dict[str, int],
    circ_type: Literal["x", "zz"],
    tfim: Optional[transverse_field_Ising_model] = None
) -> float:
    """
        Convert the counts to normalized energy.
    """
    if tfim is None:
        pbc = False
    else:
        *_, pbc = tfim.parameters
    data = []
    for bitstr in counts.keys():
        if bitstr[-1] == "0":
            # reverse the bit string (qiskit reverse convention)
            data.append(
                [int(bit) for bit in bitstr[-2::-1]] + [counts[bitstr]]
            )
    n_sys_qubits = len(list(counts.keys())[0])-1
    data = pd.DataFrame(data, columns = [
        f"q{ii}" for ii in range(n_sys_qubits)
    ] + ["counts"])
    succ_counts = data["counts"].sum()
    if succ_counts == 0:
        raise ValueError("no successful measurement")
    ienergy: int = 0

    def _unpack_query(query_res: pd.DataFrame):
        if query_res.size == 0:
            return 0
        elif query_res.size == 1:
            return query_res.to_numpy()[0,0]
        else:
            raise ValueError("duplicated keys in measurement dict")

    if circ_type == "x":
        for ii in range(n_sys_qubits):
            temp = data.groupby(f"q{ii}")[["counts"]].sum()
            for bit in range(2):
                energy_val = temp.query(f"q{ii} == {bit}")
                energy_val = _unpack_query(energy_val)
                ienergy += energy_val if bit == 0 else - energy_val
    elif circ_type == "zz":
        for ii in range(n_sys_qubits-1):
            temp = data.groupby([f"q{ii}", f"q{ii+1}"])[["counts"]].sum()
            for bits in cartesian(range(2), range(2)):
                energy_val = temp.query(
                    f"q{ii} == {bits[0]} and q{ii+1} == {bits[1]}"
                )
                energy_val = _unpack_query(energy_val)
                ienergy += energy_val if sum(bits) % 2 == 0 else - energy_val
        if pbc:
            temp = data.groupby([f"q{n_sys_qubits-1}", f"q{0}"])[["counts"]].sum()
            for bits in cartesian(range(2), range(2)):
                energy_val = temp.loc[
                    f"q{n_sys_qubits-1} == {bits[0]} and q{0} == {bits[1]}"
                ]
                energy_val = _unpack_query(energy_val)
                ienergy += energy_val if sum(bits) % 2 == 0 else - energy_val
    else:
        raise TypeError("circ_type must be either x or zz")
    energy = ienergy / succ_counts
    return energy