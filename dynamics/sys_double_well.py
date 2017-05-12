# Imports:
import numpy as np

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
    This file generates realizations of the double well (DW) dynamical system,
    within a specified time-window. The user must also define the time window
    of the sample path (trajectory), along with the hyperparameter(s).

    [Input]
    T     : Time window [t0:dt:tf].
    sig0  : System Noise (Variance).
    thet0 : Drift hyper-parameter(s).

    [Output]
    xt    : Contains the system trajectory (N x 1).

    SEE ALSO: collect_obs.

    Copyright (c) Michail D. Vrettas, PhD - November 2015.

    Last Updated: November 2015.
    """

    # Display a message.
    print(" DW - trajectory\n")

    # Get the time discretization step.
    dt = np.diff(T)[0]

    # Number of actual trajectory samples.
    N = len(T)

    # Preallocate return array for efficiency.
    x = np.zeros((N,1), dtype='float64')

    # Get the current random seed.
    r0 = np.random.get_state()

    # Set the current random seed.
    np.random.seed(6771)

    # The first value X(t=0), is chosen from the "Equilibrium Distribution":
    # x0 = 0.5*N(+mu,K) + 0.5*N(-mu,K)
    if (np.random.rand(1) > 0.5):
        x[0] = +thet0 + np.sqrt(0.5*sig0*dt)*np.random.randn(1)
    else:
        x[0] = -thet0 + np.sqrt(0.5*sig0*dt)*np.random.randn(1)

    # Noise variance coefficient.
    K = np.sqrt(sig0*dt)

    # Random variables.
    ek = np.random.randn(N,1)

    # Restore the random seed value.
    np.random.set_state(r0)

    # Create the Sample path.
    for t in range(1,N):
        x[t] = x[t-1] + 4*x[t-1]*(thet0 - x[t-1]**2)*dt + K*ek[t]

    return x

def plot_sample_path(Tk, xt):
    """
    Provides an auxiliary function to display the DW
    sample path as function of the time-window 'Tk'.
    """
    import matplotlib.pyplot as plt
    plt.plot(Tk, xt, 'ro--', linewidth=2.0)
    plt.ylabel('X(t)')
    plt.xlabel('time')
    plt.title('DW - Trajectory')
    plt.grid(True)
    plt.show()

def energy_mode(A, b, m, S, sDyn):
    """
    ENERGY MODE

    [Description]
    Energy for the double-well SDE and related quantities (including gradients).

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
       author = {Cedric Archambeau and Manfred Opper and Yuan Shen
                 and Dan Cornford and J. Shawe-Taylor},
       title = {Variational Inference for Diffusion Processes},
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

    # Observation times.
    idx = sDyn['obsX']

    # Constant value.
    c = 4.0*theta + A

    # Auxiliary constant.
    c2 = c**2

    # Iverse of noise variance.
    SigInv = 1.0/sDyn['Sig']

    # Higher order Gaussian Moments.
    Ex2 = momGauss(m,S,2)
    Ex3 = momGauss(m,S,3)
    Ex4 = momGauss(m,S,4)
    Ex6 = momGauss(m,S,6)

    # Energy from the sDyn: Eq(7)
    varQ = 16.0*Ex6 - 8.0*c*Ex4 + 8.0*b*Ex3 + c2*Ex2 - 2.0*b*c*m + b**2
    Esde = 0.5*SigInv*mytrapz(varQ, dt, idx)

    # Average drift {i.e. Eq(20) : f(t,x) = 4*x*(theta -x^2) }.
    Ef = 4.0*(theta*m - Ex3)

    # Average gradient of drift {i.e df(t,x)_dx = 4*theta - 12*x^2}.
    Edf = 4.0*(theta - 3*Ex2)

    # Derivatives of higher order Gaussian moments w.r.t. 'm' and 'S'.
    Dm2 = momGauss(m,S,2,'Dm')
    DS2 = momGauss(m,S,2,'DS')
    # ---
    Dm3 = momGauss(m,S,3,'Dm')
    DS3 = momGauss(m,S,3,'DS')
    # ---
    Dm4 = momGauss(m,S,4,'Dm')
    DS4 = momGauss(m,S,4,'DS')
    # ---
    Dm6 = momGauss(m,S,6,'Dm')
    DS6 = momGauss(m,S,6,'DS')

    # Gradients of Esde w.r.t. 'm' and 'S'.
    dEsde_dm = 0.5*SigInv*(16.0*Dm6 - 8.0*c*Dm4 + 8.0*b*Dm3 + c2*Dm2 - 2.0*b*c)
    dEsde_dS = 0.5*SigInv*(16.0*DS6 - 8.0*c*DS4 + 8.0*b*DS3 + c2*DS2)

    # Gradients of Esde w.r.t. 'Theta'.
    dEsde_dth = 4.0*SigInv*mytrapz(c*Ex2 - 4.0*Ex4 - b*m, dt, idx)

    # Gradients of Esde w.r.t. 'Sigma'.
    dEsde_dSig = -Esde*SigInv

    # --->
    return Esde, Ef, Edf, dEsde_dm, dEsde_dS, dEsde_dth, dEsde_dSig

# End-Of-File
