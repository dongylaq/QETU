# QET-U
- qet_u.py: A python package for implementing quantum eigenvalue transformation of unitary matrices with real polynomials (QET-U). 
- test_qet_u.py: Test code for the package.
- qsp_phase_matlab.py: A python interface for invoking the QSPPACK (a Matlab toolbox) for solving quantum signal processing (QSP) phase factors.
- su2_qsp_toolbox.py: A python toolbox for solving related problems of QSP in SU(2).
- circuit_to_matrix.py: A python toolbox for converting the quantum circuit (qiskit) to matrix or vector.

## Requirements
### Python packages
- python >= 3.8
- IBM Qiskit >= 0.18.3
- nptyping

### Matlab interface
- Matlab >= R2021b
The installation of "Matlab engine" is referred to https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
If running the installation listed in the instruction in Matlab commandline is troublesome, please run the code directly in the terminal.
- cd "matlabroot"/extern/engines/python
- python setup.py install
The expected output is as follows.
        running install
        running build
        running build_py
        running install_lib
        copying build/lib/matlab/engine/_arch.txt -> /Users/dongyl/opt/anaconda3/lib/python3.9/site-packages/matlab/engine
        running install_egg_info
        Removing /Users/dongyl/opt/anaconda3/lib/python3.9/site-packages/matlabengineforpython-R2021b-py3.9.egg-info
        Writing /Users/dongyl/opt/anaconda3/lib/python3.9/site-packages/matlabengineforpython-R2021b-py3.9.egg-info

## Test example
When the requirements are installed, the package can be tested by running the code.
        python test_qet_u.py
The expected output is as follows (with one figure).
        ===== test TFIM =====
        n_qubits = 4,    t_tot = 0.15453532924152408,    n_segments = 3,
        g_coupling = 1,  pbc = False,    dist = 0.1
        deg = 40
        second order Trotter error 1.09210e-03
        controlled Trotter error 1.09210e-03

        ===== test ground state preparation =====

        starting matlab engine..
        norm error = 0.0285135
        max of solution = 0.953868
        L-BFGS solver started 
        iter          obj  stepsize des_ratio
           1  +9.5874e-03 +1.00e+00 +4.85e-01
           2  +5.0994e-03 +1.00e+00 +7.03e-01
           3  +1.6589e-03 +1.00e+00 +6.68e-01
           4  +4.4775e-04 +1.00e+00 +6.92e-01
           5  +7.9723e-05 +1.00e+00 +6.70e-01
           6  +8.6696e-06 +1.00e+00 +6.33e-01
           7  +5.0660e-07 +1.00e+00 +5.82e-01
           8  +2.7697e-08 +1.00e+00 +5.68e-01
           9  +7.0012e-09 +1.00e+00 +6.70e-01
          10  +2.9404e-10 +1.00e+00 +5.64e-01
        iter          obj  stepsize des_ratio
          11  +6.6742e-12 +1.00e+00 +5.57e-01
          12  +3.7554e-14 +1.00e+00 +5.20e-01
        Stop criteria satisfied.
        succ prob = 0.171
        ||grd_state - state_circ|| = 0.08477
        E_gs                     = -4.75877
        E_recon (proj)           = -4.73961

        stopping matlab engine..
