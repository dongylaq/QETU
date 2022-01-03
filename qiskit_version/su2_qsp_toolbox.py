import numpy as np
import scipy.linalg as la
from math import ceil
from typing import Any
from nptyping import NDArray

def build_Cheby_coef_from_phi_FFT(
    phi_seq_su2: NDArray[(Any,), float]
) -> NDArray[(Any,), complex]:
    deg = len(phi_seq_su2) - 1
    n_nodes = deg + 1
    n_free_coef = ceil(n_nodes/2)
    parity = deg%2
    # Chebyshev nodes
    omega = np.pi/n_nodes * np.array(range(n_nodes))
    P = np.zeros(n_nodes, dtype=complex)
    for ii, angle in enumerate(omega):
        targ = build_QSP_matrix(phi_seq_su2, np.cos(angle))
        P[ii] = targ[0,0]
    if parity == 1:
        P = np.exp(-1j*omega) * P
    fft_c = 1/n_nodes * np.fft.fft(P)
    c = fft_c[:n_free_coef]
    if parity == 0:
        c[1:] = 2 * c[1:]
    else:
        c = 2j * c
    return c

def build_Cheby_coef_from_phi_LS(
    phi_seq_su2: NDArray[(Any,), float]
) -> NDArray[(Any,), complex]:
    deg = len(phi_seq_su2) - 1
    dof = int(np.ceil((deg+1)/2))
    omega = np.linspace(1.0/(4*dof), 0.5-1.0/(4*dof), dof) * np.pi
    chebymat = np.cos(omega.reshape(len(omega),1) * np.linspace(deg%2,deg,dof))
    P = np.zeros(omega.shape, dtype = np.complex)
    for ii, angle in enumerate(omega):
        targ = build_QSP_matrix(phi_seq_su2, np.cos(angle))
        P[ii] = targ[0,0]
    return np.linalg.solve(chebymat,P)

def build_QSP_matrix(
        phi_seq_su2: NDArray[(Any,), float],
        x: float
) -> NDArray[(2,2), complex]:
    Wx = np.array([[x, 1j*np.sqrt(1-x**2)], [1j*np.sqrt(1-x**2), x]])
    expphi = np.exp(1j*phi_seq_su2)
    ret = np.array([[expphi[0], 0], [0, np.conjugate(expphi[0])]])
    for kk in range(1,len(phi_seq_su2)):
        temp = np.array([[expphi[kk], 0], [0, np.conjugate(expphi[kk])]])
        ret = ret @ Wx @ temp
    return ret

def build_Cheby_fun_from_coef(
    x: NDArray[(Any,), float],
    coef: NDArray[(Any,), float],
    parity: int
) -> NDArray[(Any,), float]:
    # only input partial (nonzero) coefficients consist with parity
    ret = np.zeros([len(x), 1], dtype = coef.dtype)
    y = np.arccos(x.reshape(len(x),1))
    for kk in range(len(coef)):
        ret = ret + coef[kk] * np.cos((2*kk+parity)*y)
    return ret.reshape(len(ret))

def experimental_circuit_matrep(
    omega: float,
    deg: int,
    theta: float,
    varphi: float,
    chi: float
) -> NDArray[(2,2), complex]:
    def _build_su2_fsim():
        return np.array([
            [np.exp(-1j*varphi)*np.cos(theta), -1j*np.exp(1j*chi)*np.sin(theta)],
            [-1j*np.exp(-1j*chi)*np.sin(theta), np.exp(1j*varphi)*np.cos(theta)]
        ])
    def _build_su2_zrot():
        return np.array([
            [np.exp(1j*omega), 0], [0, np.exp(-1j*omega)]
        ])

    monomial = _build_su2_zrot() @ _build_su2_fsim()
    val_m, vec_m = la.eig(monomial)
    product = vec_m @ np.diag(val_m**deg) @ vec_m.conjugate().transpose()
    return product

def experimental_circuit_samples_omega(
    omega: float,
    meas_shots: int,
    deg: int,
    theta: float,
    varphi: float,
    chi: float
) -> complex:
    vec0 = np.array([1,0])
    vec_plus = 1/np.sqrt(2) * np.array([1,1])
    vec_i = 1/np.sqrt(2) * np.array([1,1j])
    temp = experimental_circuit_matrep(omega, deg, theta, varphi, chi)
    px = np.abs(vec0 @ temp @ vec_plus)**2
    py = np.abs(vec0 @ temp @ vec_i)**2
    recon_h_from_mat = px-0.5 + 1j*(py-0.5)
    if not np.isinf(meas_shots):
        sigma_x = np.sqrt((0.25 - (np.real(recon_h_from_mat)**2)) / meas_shots)
        sigma_y = np.sqrt((0.25 - (np.imag(recon_h_from_mat)**2)) / meas_shots)
        noise = sigma_x*np.random.randn() + 1j*sigma_y*np.random.randn()
        recon_h_from_mat = recon_h_from_mat + noise
    return recon_h_from_mat

def experimental_circuit_samples( 
    meas_shots: int,
    deg: int,
    theta: float,
    varphi: float,
    chi: float
) -> NDArray[(Any,), complex]:
    # Chebyshev nodes
    omega = np.pi/(2*deg-1) * np.array(range(2*deg-1))
    recon_h_from_mat = np.zeros(*omega.shape, dtype=complex)
    for ii, angle in enumerate(omega):
        recon_h_from_mat[ii] = experimental_circuit_samples_omega(
            angle, meas_shots, deg, theta, varphi, chi
        )
    return recon_h_from_mat

def matprint(
    mat: NDArray[(Any, Any), Any],
    fmt: str = "g"
) -> None:
    """
        Pretty print a numpy matrix.
    """
    col_maxes = [
        max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T
    ]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")
