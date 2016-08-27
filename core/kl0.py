# Import:
import numpy as np
from auxiliary.numerical import log_det

# Public functions:
__all__ = ['gauss']

# Listing 01:
def gauss(m0, S0, lam, Psi, sde_struct):
    '''
        GAUSSIAN LIKELIHOOD @(t=0)
    
    [Description]
    Energy for the initial state with Gaussian prior and gradients.
    
    [Input]
    m0   : initial variational mean       (D x 1).
    S0   : initial variational covariance (D x D).
    lam  : initial lagrange multiplier    (D x 1).
    Psi  : initial lagrange multiplier    (D x D).
    sde_struct : Data structure (dictionary) that holds model parameters.
    
    [Output]
    KL0      : energy of the initial state                   (1 x 1).
    dKL0_dm0 : gradient of KL0 w.r.t. the initial mean       (D x 1).
    dKL0_dS0 : gradient of KL0 w.r.t. the initial covariance (D x D).
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Updated: August 2016.
    
    Contact:
      If you find any bug or have any suggestions,
      please contact me at : <vrettasm@gmail.com>
    
    Webpage:
      The code can be downloaded from:
      <http://vrettasm.weebly.com/software.html>
    
    '''
    
    # Call the correct version.
    if (sde_struct['D'] == 1):
        return gauss_1D(m0, S0, lam, Psi, sde_struct)
    else:
        return gauss_nD(m0, S0, lam, Psi, sde_struct)

# Listing 02:
def gauss_1D(m0, S0, lam, Psi, sde_struct):
    '''
        Gaussian likelihood: 1D
    '''
    
    # Initialize prior moments.
    mu0 = sde_struct['px0']['mu0']
    tau0 = sde_struct['px0']['tau0']
    
    # Energy of the initial moment.
    z0 = m0 - mu0
    KL0 = -np.log(S0) - 0.5*(1.0 - np.log(tau0)) + 0.5/tau0*(z0**2 + S0)
    
    # Gradient w.r.t. 'm(0)'.
    dKL0_dm0 = lam[0] + z0/tau0
    
    # Gradient w.r.t. 'S(0)'.
    dKL0_dS0 = Psi[0] + 0.5*(1.0/tau0 - 1.0/S0)
    
    # --->
    return KL0, dKL0_dm0, dKL0_dS0

# Listing 03:
def gauss_nD(m0, S0, lam, Psi, sde_struct):
    '''
        Gaussian likelihood: nD
    '''
    
    # Initialize prior moments.
    mu0 = sde_struct['px0']['mu0']
    tau0 = sde_struct['px0']['tau0']
    
    # Inverse of tau0 matrix.
    tau0i = np.linalg.inv(tau0)
    
    # Inverted.
    S0i = np.linalg.inv(S0)
    
    # Energy of the initial moment.
    z0 = m0 - mu0
    KL0 = 0.5*(log_det(tau0.dot(S0i)) +\
               np.sum(np.diag(tau0i.dot(z0.T.dot(z0) + S0 - tau0))))
    
    # Gradient w.r.t. 'm(0)'.
    dKL0_dm0 = lam[0,:] + np.linalg.solve(tau0,z0.T).T
    
    # Gradient w.r.t. 'S(0)'.
    dKL0_dS0 = Psi[0,:,:] + 0.5*(tau0i - S0i)
    
    # --->
    return KL0, dKL0_dm0, dKL0_dS0

# End-Of-File
