# Numerical
import numpy as np
import numpy.random as rng

# Ploting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Variational
from core.gradEsde_mS import *
from auxiliary.numerical import *

# Public functions:
__all__ = ['system_path', 'plot_sample_path', 'energy_mode']

# Listing: 00
def lorenz63(x,u):
    """
    LORENZ63

    [Description]
    Differential equations for the Lorenz 63 system (3D)

    [Input]
    x : 3 dimensional state vector
    u : additional model parameters (Sigma, Rho, Beta)

    [Output]
    dx : return vector [3 x 1]

    Copyright (c) Michail D. Vrettas - November 2015.
    """

    # Unpack parameters. The standard values for this system are:
    # sigma = 10, rho = 28, beta = 8/3
    s, r, b = u

    # Preallocate return vector
    dx = np.zeros(x.shape, dtype='float64')

    # Differential equations
    dx[:,0] = -s*(x[:,0] - x[:,1])
    dx[:,1] = (r - x[:,2])*x[:,0] - x[:,1]
    dx[:,2] = x[:,0]*x[:,1] - b*x[:,2]

    return dx

def system_path(T, sig0, thet0):
    """
    SYSTEM PATH

    [Description]
    This file generates realizations of the stochastic Lorenz 1963 dynamical
    system, within a specified time-window. The user must also define the time
    window of the sample path (trajectory), along with the hyper-parameter(s).

    [Input]
    T     : Time window [t0:dt:tf]
    sig0  : System Noise (Variance)
    thet0 : Drift hyper-parameter(s)

    [Output]
    xt    : Contains the system trajectory (N x 3)

    SEE ALSO: collect_obs.

    Copyright (c) Michail D. Vrettas, PhD - November 2015.

    Last Updated: November 2015
    """

    # Display a message.
    print('Lorenz 1963 - trajectory')

    # Get the time discretization step.
    dt = np.diff(T)[0]

    # Number of actual trajectory samples.
    N = len(T)

    # Default starting point.
    x0 = np.array([[1.0, 1.0, 1.0]])

    # Initial conditions time step.
    dtau = 1.0e-3

    # BURN IN: Using the deterministic equations run forwards in time.
    for t in range(8000):
        x0 = x0 + lorenz63(x0,thet0)*dtau

    # Preallocate array.
    X = np.zeros((N, 3), dtype='float64')

    # Start with the new point.
    X[0] = x0

    # Noise variance coefficient.
    K = np.sqrt(sig0*dt)

    # Get the current random seed.
    # r0 = rng.get_state()

    # Set the current random seed.
    # rng.seed(6771)

    # Random variables.
    ek = rng.randn(N, 3)

    # Restore the random seed value.
    # rng.set_state(r0)

    # Create the path by solving the "stochastic" Diff.Eq. iteratively.
    for t in range(1, N):
        X[t] = X[t - 1] + lorenz63(X[t - 1, np.newaxis], thet0) * dt + ek[t].dot(K.T)

    return X

def plot_sample_path(xt):
    """
    Provides an auxiliary function to display the Lorenz 1963
    stochastic sample path in 3-Dimensional space.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], '*', label='Lorenz 3D (stochastic)')
    ax.legend()
    plt.grid(True)
    plt.show()


def energy_mode(A, b, m, S, sDyn):
    """
    ENERGY MODE

    [Description]
    Energy for the stocastic Lorenz 63 DE (3 dimensional) and related quantities
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

    Last Updated: November 2015.
    """

    # Number of discretised points
    N = sDyn['N']

    # Time discretiastion step.
    dt = sDyn['dt']

    # Inverse System Noise. Note: Since Sigma matrix is diagonal,
    # calling inv() for this inversion is accurate.
    SigInv = np.linalg.inv(sDyn['Sig'])

    # Observation times.
    idx = sDyn['obsX']

    # Diagonal elements of inverse Sigma.
    diagSigI = np.diag(SigInv)

    # Energy from the sde.
    Esde = np.zeros((N, 1), dtype='float64')

    # Average drift.
    Ef = np.zeros((N, 3), dtype='float64')

    # Average gradient of drift.
    Edf = np.zeros((N, 3, 3), dtype='float64')

    # Gradients of Esde w.r.t. 'm' and 'S'.
    dEsde_dm = np.zeros((N, 3),  dtype='float64')
    dEsde_dS = np.zeros((N, 3, 3), dtype='float64')

    # Gradients of Esde w.r.t. 'Theta'.
    dEsde_dth = np.zeros((N, 3), dtype='float64')

    # Gradients of Esde w.r.t. 'Sigma'.
    dEsde_dSig = np.zeros((N, 3), dtype='float64')

    # Drift parameters.
    theta = sDyn['theta']

    # Extract individual components.
    vS, vR, vB = theta

    # Define lambda functions:
    Fx = {
        '1': lambda x, At, bt: (lorenz63(x, theta) + x.dot(At.T) - np.tile(bt, (x.shape[0],1))) ** 2,
        '2': lambda x, _: lorenz63(x, theta)}

    # Compute the quantities iteratively.
    for t in range(N):
        # Get the values at time 't'.
        At = A[t, :, :];
        bt = b[t, :]
        St = S[t, :, :]; mt = m[t, :]

        # Compute: <(f(xt)-g(xt))'*(f(xt)-g(xt))>.
        mbar, _ = ut_approx(Fx['1'], mt, St, At, bt)

        # Esde energy: Esde(t) = 0.5*<(f(xt)-g(xt))'*SigInv*(f(xt)-g(xt))>.
        Esde[t] = 0.5 * diagSigI.dot(mbar.T)

        # Average drift: <f(Xt)>
        Ef[t, :] = np.array([
            (vS * (mt[1] - mt[0])),
            (vR * mt[0] - mt[1] - St[2,0] - mt[0] * mt[2]),
            (St[1, 0] + mt[0] * mt[1] - vB * mt[2])])

        # Average gradient of drift: <Df(Xt)>
        Edf[t, :, :] = np.array([
            [-vS, vS, 0],
            [(vR - mt[2]), -1, -mt[0]],
            [mt[1], mt[0], -vB]])

        # Approximate the expectation of the gradients.
        dmS, _ = ut_approx(gradEsde_mS, mt, St, Fx['2'], mt, St, At, bt, diagSigI)

        # Gradient w.r.t. mean mt: dEsde(t)_dmt
        dEsde_dm[t, :] = dmS[0, :3] - Esde[t] * np.linalg.solve(St, mt.T).T

        #  Gradient w.r.t. covariance St: dEsde(t)_dSt
        dEsde_dS[t, :, :] = 0.5 * (dmS[0, 3:].reshape(3, 3) - Esde[t] * np.linalg.inv(St))

        # Gradients of Esde w.r.t. 'Theta': dEsde(t)_dtheta
        dEsde_dth[t, :] = Efg_drift_theta(At, bt, mt, St, sDyn)

        # Gradients of Esde w.r.t. 'Sigma': dEsde(t)_dSigma
        dEsde_dSig[t, :] = mbar

    # Compute energy using numerical integration.
    Esde = mytrapz(Esde, dt, idx)

    # Final adjustments for the (hyper)-parameters.
    dEsde_dth = diagSigI*mytrapz(dEsde_dth, dt, idx)

    # Final adjustments for the System noise.
    dEsde_dSig = -0.5 * SigInv.dot(np.diag(mytrapz(dEsde_dSig, dt, idx))).dot(SigInv)

    return Esde, Ef, Edf, dEsde_dm, dEsde_dS, dEsde_dth, dEsde_dSig


def Efg_drift_theta(At, bt, mt, St, sDyn):
    """
    EFG_DRIFT_THETA

    [Description]
    Returns expectation : <(f-g)'*(df/dtheta)>.
    It is used when estimating the drift parameters.

    [Input]
    At  : variational linear parameter. (3 x 3)
    bt  : variational offset parameter. (1 x 3)
    mt  : marginal mean (1 x 3)
    St  : marginal covariance  (3 x 3)

    [Output]
    Gpar : gradient w.r.t. THETA (1 x 3)

    Copyright (c) Michail D. Vrettas - November 2015.
    """

    # Get the drift parameters.
    vS, vR, vB = sDyn['theta']

    # Unpack data from At.
    A11, A12, A13 = At[0]
    A21, A22, A23 = At[1]
    A31, A32, A33 = At[2]

    # Unpack data from bt.
    b1, b2, b3 = bt

    # Unpack data from mt.
    mx, my, mz = mt

    # Unpack data from St.
    # Note that this is symmetric so we extract
    # only the upper triangular elements of S(t).
    Sxx, Sxy, Sxz = St[0]
    Syy, Syz = St[1, 1:]
    Szz = St[2, 2]

    # Compute second (2nd) order expectations.
    Exx = Sxx + mx**2
    Exy = Sxy + mx*my
    Eyy = Syy + my**2
    Exz = Sxz + mx*mz
    Ezz = Szz + mz**2
    Eyz = Syz + my*mz

    # Compute third (3rd) order expectations.
    Exxz = Sxx*mz + 2 * Sxz * mx + (mx ** 2) * mz
    Exyz = Sxy*mz + Sxz * my + Syz * mx + mx * my * mz

    # Compute the expectation.
    V1 = Eyy * (vS + A12) + Exx * (vS - A11) + Exy * (A11 - 2 * vS - A12) + A13 * (Eyz - Exz) + b1 * (mx - my)
    V2 = vR * Exx - Exy - Exxz + A21 * Exx + A22 * Exy + A23 * Exz - b2 * mx
    V3 = -Exyz + vB * Ezz - A31 * Exz - A32 * Eyz - A33 * Ezz + b3 * mz

    return np.array([V1, V2, V3])
