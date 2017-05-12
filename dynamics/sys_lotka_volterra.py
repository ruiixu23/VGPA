import matplotlib.pyplot as plt
import numpy as np

from auxiliary.numerical import mytrapz

# Public functions:
__all__ = ['system_path', 'plot_sample_path', 'energy_mode']

def lotka_volterra(x, params):
    """
    LOTKA_VOLTERRA

    Differential equations for the Lotka-Volterra system (2D).
    x - 3 dimensional state vector
    params - additional parameters (alpha, beta, gamma, delta)

    [OUTPUT]
    dx - differential equation.

    Copyright (c) Ruifeng Xu - May 2017.
    """

    # Unpack parameters.
    # The standard values for this system are:
    # alpha = 2
    # beta = 1
    # delta = 4
    # gamma = 1
    alpha, beta, delta, gamma = params

    # Preallocate return vector.
    dx = np.zeros(x.shape, dtype='float64')

    # Differential equations.
    dx[0] = alpha * x[0] - beta * x[0] * x[1]
    dx[1] = delta * x[0] * x[1] - gamma * x[1]

    return dx

def system_path(T, x0, sig0, thet0):
    """
    SYSTEM_PATH

    [Description]
    This file generates realizations of the stochastic Lotka volterra dynamical
    system, within a specified time-window. The user must also define the time
    window of the sample path (trajectory), along with the hyper-parameter(s).

    [Input]
    T     : Time window [t0:dt:tf].
    x0    : The starting point.
    sig0  : System Noise (Variance).
    thet0 : Drift hyper-parameter(s).

    [Output]
    xt    : Contains the system trajectory (N x 2).

    Copyright (c) Ruifeng Xu - May 2017.

    Last Updated: May 2017.
    """

    # Display a message.
    print('Lotka-Volterra - trajectory')

    # Get the time discretization step.
    dt = np.diff(T)[0]

    # Number of actual trajectory samples.
    N = len(T)

    # Initial conditions time step.
    # dtau = 1.0e-3

    # BURN IN: Using the deterministic equations run forwards in time.
    # for t in range(8000):
        # x0 = x0 + lotka_volterra(x0, thet0) * dtau

    # Preallocate array.
    X = np.zeros((N, 2), dtype='float64')

    # Start with the new point.
    X[0] = x0

    # Noise variance coefficient.
    K = np.sqrt(sig0 * dt)

    # Get the current random seed.
    r0 = np.random.get_state()

    # Set the current random seed.
    np.random.seed(6771)

    # Random variables.
    ek = np.random.randn(N, 2)

    # Restore the random seed value.
    np.random.set_state(r0)

    # Create the path by solving the "stochastic" diff. eq. iteratively.
    for t in range(1, N):
        X[t] = X[t-1] + lotka_volterra(X[t-1], thet0) * dt + ek[t].dot(K.T)

    return X

def plot_sample_path(T, x, obs_T, obs_x):
    """
    PLOT SAMPLE PATH

    Provides an auxiliary function to display the
    stochastic sample path.
    """
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(T, x[:, 0], '-', color='r', label="x")
    ax.plot(obs_T, obs_x[:, 0], 'o', color='r', label="x")
    ax.plot(T, x[:, 1], '-', color='b', label="y")
    ax.plot(obs_T, obs_x[:, 1], 'o', color='b', label="y")
    ax.legend()
    plt.grid(True)
    plt.show()

def energy_mode(A, b, m, S, sDyn):
    """
    ENERGY_MODE

    [Description]
    Energy for the stocastic Lotka-Volterra (2 dimensional) and related quantities
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

    Copyright (c) Ruifeng Xu - May 2017.

    Last Updated: May 2017.
    """

    # Number of discretised points
    N = sDyn['N']

    # Time discretiastion step.
    dt = sDyn['dt']

    # Inverse System Noise.
    SigInv = np.linalg.inv(sDyn['Sig'])

    # Observation times.
    idx = sDyn['obsX']

    # Diagonal elements of inverse Sigma.
    diagSigI = np.diag(SigInv)

    # Energy from the sde.
    Esde = np.zeros((N, 1), dtype='float64')

    # Average drift.
    Ef = np.zeros((N, 2), dtype='float64')

    # Average gradient of drift.
    Edf = np.zeros((N, 2, 2), dtype='float64')

    # Gradients of Esde w.r.t. 'm' and 'S'.
    dEsde_dm = np.zeros((N, 2),  dtype='float64')
    dEsde_dS = np.zeros((N, 2, 2),dtype='float64')

    # Gradients of Esde w.r.t. 'Theta'.
    dEsde_dth = np.zeros((N, 2), dtype='float64')

    # Gradients of Esde w.r.t. 'Sigma'.
    dEsde_dSig = np.zeros((N, 2), dtype='float64')

    # Drift parameters.
    vAlpha, vBeta, vDelta, vGamma = sDyn['theta']

    # Compute the quantities iteratively.
    for t in range(N):
        # Get the values at time 't'.
        At = A[t, :, :]
        bt = b[t, :]
        St = S[t, :, :]
        mt = m[t, :]

        # Compute the energy and the related gradients.
        Efg, Edm, EdS = Energy_dm_dS(At, bt, mt, St, diagSigI, sDyn)

        # Energy Esde(t):
        Esde[t] = 0.5 * diagSigI.dot(Efg)

        # Gradient dEsde(t)/dm(t):
        dEsde_dm[t, :] = Edm

        # Gradient dEsde(t)/dS(t):
        dEsde_dS[t, :, :] = EdS

        # Average drift: <f(Xt)>
        Ef[t, :] =  np.array([
            (vS * (mt[1] - mt[0])),
            (vR * mt[0] - mt[1] - St[2, 0] - mt[0] * mt[2]),
            (St[1, 0] + mt[0] * mt[1] - vB * mt[2])
        ])

        # Average gradient of drift: <Df(Xt)>
        Edf[t, :, :] = np.array([
            [-vS, vS, 0],
            [(vR - mt[2]), -1, -mt[0]],
            [mt[1], mt[0], -vB]
        ])

        # Gradients of Esde w.r.t. 'Theta'.
        dEsde_dth[t, :] = Efg_drift_theta(At, bt, mt, St, sDyn)

        # Gradients of Esde w.r.t. 'Sigma'.
        dEsde_dSig[t, :] = Efg

    # Compute energy using numerical integration.
    Esde = mytrapz(Esde, dt, idx)

    # Final adjustments for the (hyper)-parameters.
    dEsde_dth = diagSigI*mytrapz(dEsde_dth, dt, idx)

    # Final adjustments for the System noise.
    dEsde_dSig = -0.5*SigInv.dot(np.diag(mytrapz(dEsde_dSig, dt, idx))).dot(SigInv)

    return Esde, Ef, Edf, dEsde_dm, dEsde_dS, dEsde_dth, dEsde_dSig

def Energy_dm_dS(At, bt, mt, St, diagSigI, sDyn):
    """
    ENERGY_DM_DS

    [Description]
    Returns the Energy of the Lotka-Volterra system and related gradients.
    More specifically, it returns the gradient of the Esde(t) with
    respect to the marginal mean m(t) and the marginal covariance S(t).

    [Input]
    At  : variational linear parameter. (D x D).
    bt  : variational offset parameter. (1 x D).
    mt  : marginal mean (1 x D).
    St  : marginal covariance  (D x D).
    diagSigI : diagonal elements of inverted system noise covariance (1 x D)

    [Output]
    Efg      : Expectation : <(f-g)(f-g)'> (1 x D).
    dEsde_dm : dEsde(t)/dm(t) (1 x D).
    dEsde_dS : dEsde(t)/dS(t) (D x D).

    Copyright (c) Ruifeng Xu - May 2017.

    Last Updated: May 2017.
    """

    # Get the drift parameters.
    vAlpha, vBeta, vDelta, vGamma = sDyn['theta']

    # Unpack data from At.
    A11, A12, A13, A14 = At[0]
    A21, A22, A23, A24 = At[1]
    A31, A32, A33, A34 = At[2]
    A41, A42, A43, A44 = At[2]

    # Unpack data from bt.
    b1, b2, b3, b4 = bt

    # Unpack data from mt.
    mx, my = mt

    # Unpack data from St.
    # Note that this is symmetric so we extract
    # only the upper triangular elements of S(t).
    Sxx, Sxy = St[0]
    Syy = St[1, 1]

    # Compute second (2nd) order expectations.
    Exx = Sxx + mx ** 2
    Exy = Sxy + mx * my
    Eyy = Syy + my ** 2

    # Compute third (3rd) order expectations.
    Exxy = Sxx * my + 2 * Sxy * mx + (mx ** 2) * my
    Exyy = Syy * mx + 2 * Sxy * my + (my ** 2) * mx

    # Compute forth (4th) order expectations.
    Exxyy = Sxx * (my ** 2 + Syy) + Syy * (mx ** 2) + 4 * Sxy * mx * my + (mx * my) ** 2 + 2 * (Sxy ** 2)

    # Compute the expectation for the Energy.
    EX = (vS ** 2) * (Eyy + Exx - 2 * Exy) + (A11 ** 2) * Exx + (A12 ** 2) * Eyy + \
         (A13 ** 2) * Ezz + b1 ** 2 + 2 * (A11 * A12 * Exy + A11 * A13 * Exz - b1 * A11 * mx + \
          A12 * A13 * Eyz - b1 * A12 * my - b1 * A13 * mz + vS * (A11 * Exy + A12 * Eyy + \
          A13 * Eyz - b1 * my - A11 * Exx - A12 * Exy - A13 * Exz + b1 * mx))
    EY = (vR ** 2) * Exx + Eyy + Exxzz + (A21 ** 2) * Exx + (A22 ** 2) * Eyy + \
         (A23 ** 2) * Ezz + b2 ** 2 + 2 * (Exyz - A21 * Exy - A22 * Eyy - A23 * Eyz - \
          A21 * Exxz - A22 * Exyz - A23 * Exzz + A21 * A22 * Exy + A21 * A23 * Exz + \
          A22 * A23 * Eyz - vR * (Exy + Exxz - A21 * Exx - A22 * Exy - A23 * Exz) - \
          b2 * (vR * mx - my - Exz + A21 * mx + A22 * my + A23 * mz))

    # Expectation of the distance between the drift
    # and the linear approximation : <(f-g)*(f-g)'>.
    Efg = np.array([EX, EY])

    # Compute the derivatives of second (2nd) order
    # expectations with respect to mt.
    dExx_dmx = 2*mx
    dExy_dmx = my

    dEyy_dmy = 2*my
    dExy_dmy = mx

    # Compute the derivatives of second (2nd) order
    # expectations with respect to St.
    dExx_dSxx = 1
    dEyy_dSyy = 1
    dExy_dSxy = 1

    # Compute the derivatives of third (3rd) order expectations with respect to mt.
    dExxy_dmx = 2*Exy
    dExyy_dmx = Eyy

    dExxy_dmy = Exx
    dExyy_dmy = 2*Exy

    # Compute the derivatives of third (3rd) order expectations with respect to St.
    dExxy_dSxx = my
    dExxy_dSxy = 2*mx
    dExyy_dSxy = 2*my
    dExyy_dSyy = mx

    # Compute the derivatives of forth (4th) order expectations w.r.t. to mt.
    dExxyy_dmx = 2*Exyy
    dExxyy_dmy = 2*Exxy

    # Compute the derivatives of forth (4th) order expectations w.r.t. to St.
    dExxyy_dSxx = Eyy
    dExxyy_dSxy = 4*Exy
    dExxyy_dSyy = Exx

    # Compute the expectation for the dEsde(t)/dm(t).
    dmx1 = dExx_dmx*(vS**2 + A11**2) + 2*(dExy_dmx*(-vS**2 + vS*A11 - vS*A12 + \
           A11*A12) + dExz_dmx*(A11 - vS)*A13 - vS*A11*dExx_dmx + b1*(vS - A11))
    dmx2 = dExxzz_dmx + dExx_dmx*(vR**2 + A21**2) + \
           2*(dExy_dmx*(-vR + vR*A22 - A21 + A21*A22) + dExz_dmx*(vR*A23 + b2 + \
           A21*A23) + dExyz_dmx*(1 - A22) - vR*dExxz_dmx + vR*A21*dExx_dmx - \
           A21*dExxz_dmx - A23*dExzz_dmx - b2*(vR + A21))

    dmy1 = dEyy_dmy*(vS**2 + A12**2) + 2*(dExy_dmy*(-(vS**2) + vS*A11 - \
           vS*A12 + A11*A12) + dEyz_dmy*(vS + A12)*A13 + vS*A12*dEyy_dmy - \
           b1*(vS+A12))
    dmy2 = dEyy_dmy*(1 + A22**2) + 2*(dExy_dmy*(-vR + vR*A22 - A21 + A21*A22) + \
           dExyz_dmy*(1 - A22) - A22*dEyy_dmy + dEyz_dmy*(A22*A23 - A23) + \
           b2*(1-A22))

    # Gradient of the energy with respect to the marginal mean.
    dEsde_dm = 0.5*np.array([[dmx1, dmx2], [dmy1, dmy2]]).dot(diagSigI)

    # Take the diagonal elements.
    iSx, iSy = diagSigI

    # Compute the expectation for the dEsde(t)/dS(t).
    dSxx = iSx*((vS - A11)**2)*dExx_dSxx + \
           iSy*(dExxzz_dSxx + dExx_dSxx*((vR + A21)**2) - 2*dExxz_dSxx*(vR + A21))+ \
           iSz*(dExxyy_dSxx + (A31**2)*dExx_dSxx + 2*A31*dExxy_dSxx)
    dSxy = iSx*2*(vS*A11 - vS**2 - vS*A12 + A11*A12)*dExy_dSxy + \
           iSy*2*(dExy_dSxy*(vR*A22 - vR - A21 + A21*A22) + dExyz_dSxy*(1 - A22)) + \
           iSz*(dExxyy_dSxy + 2*(dExyz_dSxy*(A33 - vB) + A31*dExxy_dSxy + \
           A32*dExyy_dSxy + dExy_dSxy*(A31*A32 - b3)))
    dSyy = iSx*((vS + A12)**2)*dEyy_dSyy + iSy*((1 - A22)**2)*dEyy_dSyy + \
           iSz*(dExxyy_dSyy + (A32**2)*dEyy_dSyy + 2*A32*dExyy_dSyy)

    # Gradient of the energy with respect to the marginal covariance.
    dEsde_dS = 0.5*np.array([[dSxx, dSxy], [dSxy, dSyy])
    return Efg, dEsde_dm, dEsde_dS

def Efg_drift_theta(At, bt, mt, St, sDyn):
    """
    EFG_DRIFT_THETA

    [Description]
    Returns expectation : <(f-g)'*(df/dtheta)>.
    It is used when estimating the drift parameters.

    [Input parameters]
    At  : variational linear parameter. (3 x 3).
    bt  : variational offset parameter. (1 x 3).
    mt  : marginal mean (1 x 3).
    St  : marginal covariance  (3 x 3).

    [Output parameters]
    Gpar : gradient w.r.t. THETA (1 x 3).

    Copyright (c) Michail D. Vrettas - November 2015.
    """

    # Get the drift parameters.
    vAlpha, vBeta, vDelta, vGamma = sDyn['theta']


    # Unpack data from At.
    A11, A12, A13 = At[0]
    A21, A22, A23 = At[1]
    A31, A32, A33 = At[2]

    # Unpack data from bt.
    b1, b2, b3 = bt

    # Unpack data from mt.
    mx, my, mz = mt

    # Unpack data from St.
    # Note that this is symmetric so we extract only the upper triangular
    # elements of S(t).
    Sxx, Sxy, Sxz = St[0]
    Syy, Syz = St[1,1:]
    Szz = St[2,2]

    # Compute second (2nd) order expectations.
    Exx = Sxx + mx**2
    Exy = Sxy + mx*my
    Eyy = Syy + my**2
    Exz = Sxz + mx*mz
    Ezz = Szz + mz**2
    Eyz = Syz + my*mz

    # Compute third (3rd) order expectations.
    Exxz = Sxx*mz + 2*Sxz*mx + (mx**2)*mz
    Exyz = Sxy*mz + Sxz*my + Syz*mx + mx*my*mz

    # Compute the expectation.
    V1 = Eyy*(vS + A12) + Exx*(vS - A11) + Exy*(A11 - 2*vS - A12) + \
         A13*(Eyz - Exz) + b1*(mx - my)
    V2 = vR*Exx - Exy - Exxz + A21*Exx + A22*Exy + A23*Exz - b2*mx
    V3 = -Exyz + vB*Ezz - A31*Exz - A32*Eyz - A33*Ezz + b3*mz

    return np.array([V1, V2, V3])
