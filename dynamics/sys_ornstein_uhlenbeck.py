# Imports:
import numpy as np
import numpy.random as rng

# Variational:
from auxiliary.momGauss import momGauss

# "Numerical/auxiliary" (local):
from auxiliary.numerical import mytrapz

# Public functions:
__all__ = ['system_path', 'plot_sample_path', 'energy_mode']

# Listing: 01
def system_path(T, sig0, thet0):
    """ 
        SYSTEM PATH
    
    [Description]
    This file generates realizations of the Ornstein-Uhlenbeck (OU) dynamical
    system, within a specified time-window. The user must also define the time
    window of the sample path (trajectory), along with the hyperparameter(s).
    
    [Input]
    T     : Time window [t0:dt:tf].
    sig0  : System Noise (Variance).
    thet0 : Drift hyper-parameter(s).
    
    [Output]
    xt    : Contains the system trajectory (N x 1).
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
     
    Last Updated: November 2015.
    """
    
    # Display a message.
    print(" OU - trajectory\n")
    
    # Get the time discretization step.
    dt = np.diff(T)[0]
    
    # Number of actual trajectory samples.
    N = len(T)
    
    # Preallocate return array for efficiency.
    x = np.zeros((N,1), dtype='float64')
    
    # Default mean value is zero.
    mu = 0
    
    # Noise variance coefficient.
    K = np.sqrt(sig0*dt)
    
    # The first value X(t=0) = 0 or X(t=0) ~ N(mu,K)
    x[0] = mu
    
    # Random variables.
    ek = rng.randn(N,1)
    
    # Create the Sample path.
    for t in range(1,N):
        x[t] = x[t-1] + thet0*(mu - x[t-1])*dt + K*ek[t]
    
    # --->
    return x

# Listing: 02
def plot_sample_path(Tk, xt):
    """
    Provides an auxiliary function to display the DW
    sample path as function of the time-window 'Tk'.
    """
    import matplotlib.pyplot as plt
    plt.plot(Tk, xt, 'ro--', linewidth=2.0)
    plt.ylabel('X(t)')
    plt.xlabel('time')
    plt.title('OU - Trajectory')
    plt.grid(True)
    plt.show()

# Listing: 03
def energy_mode(A, b, m, S, sDyn):
    """
        ENERGY MODE
    
    [Description]
    Energy for the double-well SDE, and related quantities (including gradients).
    
    [Input]
    A         : variational linear parameters (N x 1).
    b         : variational offset parameters (N x 1).
    m         : narginal means (N x 1).
    S         : marginal variances  (N x 1).
    sDyn      : structure containing additional parameters.
    
    [Output]
    Esde      : total energy of the sde.
    Ef        : average drift (N x 1).
    Edf       : average differentiated drift (N x 1).
    dEsde_dm  : gradient of Esde w.r.t. the means (N x 1).
    dEsde_dS  : gradient of Esde w.r.t. the covariance (N x 1).
    dEsde_dth : gradient of Esde w.r.t. the parameter theta.
    dEsde_dSig: gradient of Esde w.r.t. the parameter Sigma.
        
    NOTE: The equation numbers correspond to the paper:
    
    @CONFERENCE{Archambeau2007b,
       author = {Cedric Archambeau, Manfred Opper, Yuan Shen, Dan Cornford and
       J. Shawe-Taylor},title = {Variational Inference for Diffusion Processes},
       booktitle = {Annual Conference on Neural Information Processing Systems},
       year = {2007}
    }
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Updated: November 2015.
    """
    
    # Find the time step.
    dt = sDyn['dt']
    
    # Extract the drift parameter.
    theta = sDyn['theta']
    
    # Inverse noise variance.
    SigInv = 1.0/sDyn['Sig']
    
    # Observation times.
    idx = sDyn['obsX']
    
    # Higher order Gaussian Moments.
    Ex2 = momGauss(m,S,2)
    
    # Precompute these quantities only once.
    Q1 = (theta - A)**2; Q2 = A*b
    
    # Energy from the sDyn: Eq(7)
    varQ = Ex2*Q1 + 2*m*(theta - A)*b + b**2
    Esde = 0.5*SigInv*mytrapz(varQ, dt, idx)
    
    # Average drift.
    Ef = -theta*m
    
    # Average gradient of drift.
    Edf = -theta*np.ones(m.shape)
    
    # Gradients of Esde w.r.t. 'm' and 'S'.
    dEsde_dm = SigInv*(m*(theta - A)**2 + theta*b - Q2)
    dEsde_dS = 0.5*SigInv*Q1
    
    # Gradients of Esde w.r.t. 'Theta'.
    dEsde_dth = SigInv*mytrapz(Ex2*(theta-A) + m*b, dt, idx)
    
    # Gradients of Esde w.r.t. 'Sigma'.
    dEsde_dSig = -SigInv*Esde
    
    # --->
    return Esde, Ef, Edf, dEsde_dm, dEsde_dS, dEsde_dth, dEsde_dSig

# End-Of-File
