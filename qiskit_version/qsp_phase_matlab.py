"""Evaluate the phase factors by interfacing with qsppack

Interface with matlab to run cvx_qsp and qsppack directly to obtain
the approximate polynomial and phase factors.

Last revision: 01/09/2022
"""

import numpy as np


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


def sigma_to_lambda(x):
    return np.arccos(2.0 * x**2 - 1.0)
        
def lambda_to_sigma(x):
    return np.sqrt((1.0 + np.cos(x)) / 2)


class QSPPhase(object):
    """Evaluate the phase factors by interfacing with qsppack"""

    def __init__(self):
        print("\nstarting matlab engine..")
        self.eng = matlab.engine.start_matlab()
        self.opts = {
                'npts': 400, 
                'epsil': 0.01, 
                'fscale': 0.9, 
                'criteria': 1e-6,
                } 

    def cvx_qsp_heaviside(self, deg, E_min, E_mu_m, E_mu_p, E_max, verbose=False):
        """ Evaluate the phase factors approximating a Heaviside
        function of degree deg, which is approximately 1 on 
        [E_min, E_mu_m], and approximately 0 on [E_mu_p, E_max]. 

        Here 
            0 < E_min < E_mu_m < E_mu_p < E_max < pi
        """
        sigma_min = lambda_to_sigma(E_max)
        sigma_mu_m = lambda_to_sigma(E_mu_p)
        sigma_mu_p = lambda_to_sigma(E_mu_m)
        sigma_max = lambda_to_sigma(E_min)
        

        phi_seq_matlab = self.eng.cvx_qsp_heaviside(int(deg), 
                float(sigma_min), float(sigma_mu_m), 
                float(sigma_mu_p), float(sigma_max), 
                self.opts['npts'], self.opts['epsil'], self.opts['fscale'], 
                self.opts['criteria'], int(verbose))

        phi_seq_su2 = np.array(phi_seq_matlab).reshape(-1)

        return phi_seq_su2

    def __del__(self):
        print("\nstopping matlab engine..")
        self.eng.quit()
