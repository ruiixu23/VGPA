# Numerical
import numpy as np

# Variational
from core.gradEsde_mS import *
from auxiliary.numerical import *

# Public functions
__all__ = ['system_path', 'plot_sample_path', 'energy_mode']

# Generates the shifted state vectors
f1 = lambda x: np.roll(x,-1)
b1 = lambda x: np.roll(x,+1)
b2 = lambda x: np.roll(x,+2)

def lorenz96(x, f):
    """
    LORENZE96

    [Description]
    Differential equations for the Lorenz 96 system.
    x - State vector
    f - drift parameter

    [OUTPUT]
    dx - return vector.

    Copyright (c) Michail D. Vrettas - November 2015.
    """

    # Preallocate return vector.
    dx = np.zeros(x.shape, dtype='float64')

    # Differential equations.
    for i in range(x.shape[0]):
        xi = x[i, :]
        dx[i, :] = (f1(xi) - b2(xi)) * b1(xi) - xi + f

    return dx

def system_path(T, D, sig0, thet0):
    """
    SYSTEM PATH

    [Description]
    This file generates realizations of the stochastic Lorenz 1996 dynamical
    system, within a specified time-window. The user must also define the time
    window of the sample path (trajectory), along with the hyperparameter(s).

    [Input]
    T     : Time window [t0:dt:tf].
    D     : The number of states.
    sig0  : System Noise (Variance).
    thet0 : Drift hyper-parameter(s).

    [Output]
    xt    : Contains the system trajectory (N x D).

    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    """

    # Display a message
    print('Lorenz 1996 - trajectory\n')

    # Get the time discretization step.
    dt = np.diff(T)[0]

    # Number of actual trajectory samples.
    N = len(T)

    # Default starting point.
    x0 = np.array([[thet0] * D], dtype='float64')

    # Purturb the L-th dimension by 1/1000.
    x0[0, int(D / 2)] += 1.0e-3

    # Time-step for initial discretisation.
    dtau = 1.0e-3

    # BURN IN: Using the deterministic equations run forwards in time.
    for t in range(50000):
        x0 = x0 + lorenz96(x0, thet0) * dtau

    # Preallocate array.
    X = np.zeros((N, D), dtype='float64')

    # Start with the new point.
    X[0, :] = x0

    # Noise variance coefficient.
    K = np.sqrt(sig0 * dt)

    # Get the current random seed.
    r0 = np.random.get_state()

    # Set the current random seed.
    np.random.seed(6771)

    # Random variables.
    ek = np.random.randn(N, D)

    # Create the path by solving the SDE iteratively.
    for t in range(1, N):
        X[t] = X[t-1] + lorenz96(X[t-1,np.newaxis], thet0) * dt + ek[t].dot(K.T)

    # Restore the random seed value.
    np.random.set_state(r0)

    return X

def plot_sample_path(xt):
    """
    Provides an auxiliary function to display the Lorenz'96 stochastic path.
    """
    # Plotting:
    from matplotlib.pyplot import figure, show

    # Create a new figure.
    fig = figure()

    ax = fig.add_subplot(111)
    ax.imshow(xt.T, aspect='auto')

    ax.set_title('Lorenz 1996')
    show()

def energy_mode(A, b, m, S, sDyn):
    """
     ENERGY_MODE

    [Description]
    Energy for the stocastic Lorenz 1996 System and related quantities
    (including gradients).

    [Input]
    A         : variational linear parameters (N x D x D).
    b         : variational offset parameters (N x D).
    m         : narginal means (N x D).
    S         : marginal variances  (N x D x D).
    sDyn      : structure containing additional parameters.

    [Output]
    Esde      : total energy of the sde.
    Ef        : average drift (N x D).
    Edf       : average differentiated drift (N x D).
    dEsde_dm  : gradient of Esde w.r.t. the means (N x D).
    dEsde_dS  : gradient of Esde w.r.t. the covariance (N x D x D).
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
    """

    # Number of discretised points
    N = sDyn['N']

    # Number of states
    D = sDyn['D']

    # Time discretiastion step
    dt = sDyn['dt']

    # System noise
    Sig = sDyn['Sig']

    # Drift (forcing) parameter
    theta = sDyn['theta']

    # Inverse System Noise
    SigInv = np.linalg.inv(Sig)

    # Observation times
    idx = sDyn['obsX']

    # Diagonal elements of inverse Sigma
    diagSigI = np.diag(SigInv)

    # Energy from the sDyn
    Esde = np.zeros((N, 1), dtype='float64')

    # Average drift
    Ef = np.zeros((N, D), dtype='float64')

    # Average gradient of drift
    Edf = np.zeros((N, D, D), dtype='float64')

    # Gradients of Esde w.r.t. 'm' and 'S'
    dEsde_dm = np.zeros((N, D), dtype='float64')
    dEsde_dS = np.zeros((N, D, D),dtype='float64')

    # Gradients of Esde w.r.t. 'Theta'
    dEsde_dth = np.zeros((N, D), dtype='float64')

    # Gradients of Esde w.r.t. 'Sigma'
    dEsde_dSig = np.zeros((N, D), dtype='float64')

    # Define lambda functions
    Fx = {
        '1': lambda x, At, bt: (lorenz96(x, theta) + x.dot(At.T) - np.tile(bt, (x.shape[0],1))) ** 2,
        '2': lambda x, _: lorenz96(x, theta)}

    # Compute the quantities iteratively
    for t in range(N):
        # Get the values at time 't'
        At = A[t, :, :]; bt = b[t, :]
        St = S[t, :, :]; mt = m[t, :]

        # Compute: <(f(xt)-g(xt))'*(f(xt)-g(xt))>.
        mbar, _ = ut_approx(Fx['1'], mt, St, At, bt)

        # Esde energy: Esde(t) = 0.5*<(f(xt)-g(xt))'*SigInv*(f(xt)-g(xt))>.
        Esde[t] = 0.5 * diagSigI.dot(mbar.T)

        # Average drift: <f(Xt)>
        Ef[t,:] = E_L96_drift(mt, St, theta, D)

        # Average gradient of drift: <Df(Xt)>
        Edf[t, :, :] = E_L96_drift_dx(mt, D)

        # Approximate the expectation of the gradients.
        dmS, _ = ut_approx(gradEsde_mS, mt, St, Fx['2'], mt, St, At, bt, diagSigI)

        # Gradient w.r.t. mean mt: dEsde(t)_dmt
        dEsde_dm[t, :] = dmS[0, :D] - Esde[t] * np.linalg.solve(St,mt.T).T

        #  Gradient w.r.t. covariance St: dEsde(t)_dSt
        dEsde_dS[t, :, :] = 0.5 * (dmS[0, D:].reshape(D, D) - Esde[t]*np.linalg.inv(St))

        # Gradients of Esde w.r.t. 'Theta': dEsde(t)_dtheta
        dEsde_dth[t, :] = Ef[t, :] + mt.dot(At.T) - bt

        # Gradients of Esde w.r.t. 'Sigma': dEsde(t)_dSigma
        dEsde_dSig[t, :] = mbar

    # Compute energy using numerical integration.
    Esde = mytrapz(Esde, dt, idx)

    # Final adjustments for the (hyper)parameters.
    dEsde_dth = diagSigI*mytrapz(dEsde_dth, dt, idx)

    # Final adjustments for the System noise.
    dEsde_dSig = -0.5 * SigInv.dot(np.diag(mytrapz(dEsde_dSig, dt, idx))).dot(SigInv)

    return Esde, Ef, Edf, dEsde_dm, dEsde_dS, dEsde_dth, dEsde_dSig

def E_L96_drift(mt, St, theta, D):
    """
    E_L96_DRIFT

    [Description]
    Returns the mean value of the drift function <f(x)>.

    [INPUT PARAMETERS]
    mt    : mean vector (1 x D)
    St    : covariance matrix (D x D)
    theta : drift parameter.

    [OUTPUT PARAMETERS]
    EF : mean of the drift function (1 x D).

    Copyright (c) Michail D. Vrettas - November 2015.
    """

    # Preallocate vector.
    Cxx = np.array([0] * D, dtype='float64')

    # Local index array: [0, 1, 2, ... , 39]
    idx = np.arange(0, D)

    # Get access to the covariances at the desired points.
    for i in range(D):
        Cxx[i] = St[f1(idx)[i],b1(idx)[i]] - St[b2(idx)[i],b1(idx)[i]]

    # Compute the expected value.
    EF = Cxx + (f1(mt)-b2(mt))*b1(mt) - mt + theta

    return EF

def E_L96_drift_dx(x, D):
    """
    E_L96_DRIFT_dX

    [Description]
    Returns the mean value of the gradient of the drift function
    with respect to the state vector: <df(x)/dx>.

    [INPUT PARAMETERS]
    x  : input state samples (D x 1).

    [OUTPUT PARAMETERS]
    Ex : mean gradient w.r.t. to X (D x D).

    Copyright (c) Michail D. Vrettas - June 2009.
    """

    # Preallocate return matrix.
    Ex = np.zeros((D, D), dtype='float64')

    # Local index array: [0, 1, 2, ... , 39]
    idx = np.arange(0, D)

    # Compute the gradient of the state vector at each time point.
    for i in range(D):
        # Generate zeros
        Gx = [0] * D

        # Compute the i-th ODE gradient.
        Gx[i] = -1
        Gx[f1(idx)[i]] = +b1(x)[i]
        Gx[b2(idx)[i]] = -b1(x)[i]
        Gx[b1(idx)[i]] = +f1(x)[i] - b2(x)[i]

        # Store i-th gradient.
        Ex[i,:] = Gx

    return Ex
