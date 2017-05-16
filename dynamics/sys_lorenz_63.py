# Import only what is necessary
import numpy as np
import numpy.random as rng

# "Numerical/auxiliary" (local)
from auxiliary.numerical import mytrapz

# Public functions
__all__ = ['system_path', 'plot_sample_path', 'energy_mode']


def lorenz63(x, u):
    """
    Lorenz63(x,u)

    Differential equations for the Lorenz 63 system (3D).
    x - 3 dimensional state vector
    u - additional parameters (sigma, rho, beta)

    [OUTPUT]
    dx - return vector.

    Copyright (c) Michail D. Vrettas - November 2015.
    """

    # Unpack parameters. The standard values for this system are:
    # sigma = 10, rho = 28, beta = 8/3
    s, r, b = u

    # Preallocate return vector.
    dx = np.zeros(x.shape, dtype='float64')

    # Differential equations.
    dx[0] = -s * (x[0] - x[1])
    dx[1] = (r - x[2]) * x[0] - x[1]
    dx[2] = x[0] * x[1] - b * x[2]

    # --->
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
    x0 = np.array([1.0, 1.0, 1.0])

    # Initial conditions time step.
    dtau = 1.0e-3

    # BURN IN: Using the deterministic equations run forwards in time.
    for t in range(8000):
        x0 = x0 + lorenz63(x0, thet0) * dtau

    # Preallocate array.
    X = np.zeros((N, 3), dtype='float64')

    # Start with the new point.
    X[0] = x0

    # Noise variance coefficient.
    K = np.sqrt(sig0 * dt)

    # Get the current random seed.
    r0 = rng.get_state()

    # Set the current random seed.
    rng.seed(6771)

    # Random variables.
    ek = rng.randn(N, 3)

    # Restore the random seed value.
    rng.set_state(r0)

    # Create the path by solving the "stochastic" Diff.Eq. iteratively.
    for t in range(1, N):
        X[t] = X[t - 1] + lorenz63(X[t-1], thet0) * dt + ek[t].dot(K.T)

    # --->
    return X


def plot_sample_path(xt):
    """
    Provides an auxiliary function to display the Lorenz'63
    stochastic sample path in 3-Dimensional space.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], '*', label='Lorenz 3D (stochastic)')
    ax.legend()
    plt.grid(True)
    plt.show()


def energy_mode(A, b, m, S, sDyn):
    """
    ENERGY MODE:

    [Description]
    Energy for the stochastic Lorenz 63 DE (3 dimensional) and related quantities
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

    # {N}umber of discretised points
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
    vS, vR, vB = sDyn['theta']

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
        Ef[t, :] = np.array([(vS*(mt[1] - mt[0])),
                             (vR*mt[0] - mt[1] - St[2, 0] - mt[0] * mt[2]),
                             (St[1, 0] + mt[0] * mt[1] - vB * mt[2])])

        # Average gradient of drift: <Df(Xt)>
        Edf[t, :, :] = np.array([[-vS, vS, 0], [(vR - mt[2]), -1, -mt[0]], [mt[1], mt[0], -vB]])

        # Gradients of Esde w.r.t. 'Theta'.
        dEsde_dth[t, :] = Efg_drift_theta(At, bt, mt, St, sDyn)

        # Gradients of Esde w.r.t. 'Sigma'.
        dEsde_dSig[t, :] = Efg

    # Compute energy using numerical integration.
    Esde = mytrapz(Esde, dt, idx)

    # Final adjustments for the (hyper)-parameters.
    dEsde_dth = diagSigI * mytrapz(dEsde_dth, dt, idx)

    # Final adjustments for the System noise.
    dEsde_dSig = -0.5 * SigInv.dot(np.diag(mytrapz(dEsde_dSig, dt, idx))).dot(SigInv)

    return Esde, Ef, Edf, dEsde_dm, dEsde_dS, dEsde_dth, dEsde_dSig


def Energy_dm_dS(At, bt, mt, St, diagSigI, sDyn):
    """
    ENERGY_DM_DS

    [Description
    Returns the Energy of the Lorenz 3D system and related gradients.
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

    Copyright (c) Michail D. Vrettas - November 2015.

    Last Updated: November 2015.
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
    Exz = Sxz + mx*mz
    Eyy = Syy + my**2
    Eyz = Syz + my*mz
    Ezz = Szz + mz**2

    # Compute third (3rd) order expectations.
    Exxy = Sxx*my + 2*Sxy*mx + (mx**2)*my
    Exxz = Sxx*mz + 2*Sxz*mx + (mx**2)*mz
    Exyy = Syy*mx + 2*Sxy*my + (my**2)*mx
    Exzz = Szz*mx + 2*Sxz*mz + (mz**2)*mx
    Exyz = Sxy*mz + Sxz*my + Syz*mx + mx*my*mz

    # Compute forth (4th) order expectations.
    Exxyy = Sxx*(my**2 + Syy) + Syy*(mx**2) + 4*Sxy*mx*my + (mx*my)**2 + 2*(Sxy**2)
    Exxzz = Sxx*(mz**2 + Szz) + Szz*(mx**2) + 4*Sxz*mx*mz + (mx*mz)**2 + 2*(Sxz**2)

    # Compute the expectation for the Energy.
    EX = (
        (vS**2)*(Eyy + Exx - 2*Exy) + (A11**2)*Exx + (A12**2)*Eyy + (A13**2)*Ezz + b1**2 +
        2*(A11*A12*Exy + A11*A13*Exz - b1*A11*mx + A12*A13*Eyz - b1*A12*my - b1*A13*mz +
           vS*(A11*Exy + A12*Eyy + A13*Eyz - b1*my - A11*Exx - A12*Exy - A13*Exz + b1*mx))
    )
    EY = (
        (vR**2)*Exx + Eyy + Exxzz + (A21**2)*Exx + (A22**2)*Eyy + (A23**2)*Ezz + b2**2 +
        2*(Exyz - A21*Exy - A22*Eyy - A23*Eyz - A21*Exxz - A22*Exyz - A23*Exzz + A21*A22*Exy + A21*A23*Exz +
           A22*A23*Eyz - vR*(Exy + Exxz - A21*Exx - A22*Exy - A23*Exz) -
           b2*(vR*mx - my - Exz + A21*mx + A22*my + A23*mz))
    )
    EZ = (
        Exxyy + (vB**2)*Ezz + (A31**2)*Exx + (A32**2)*Eyy + (A33**2)*Ezz + b3**2 +
        2*(A31*Exxy + A32*Exyy + A33*Exyz + A31*A32*Exy + A31*A33*Exz + A32*A33*Eyz -
           vB*(Exyz + A31*Exz + A32*Eyz + A33*Ezz) - b3*(Exy - vB*mz + A31*mx + A32*my + A33*mz))
    )

    # Expectation of the distance between the drift
    # and the linear approximation : <(f-g)*(f-g)'>.
    Efg = np.array([EX, EY, EZ])

    # Compute the derivatives of second (2nd) order
    # expectations with respect to mt.
    dExx_dmx = 2 * mx
    dExy_dmx = my
    dExz_dmx = mz

    dEyy_dmy = 2 * my
    dExy_dmy = mx
    dEyz_dmy = mz

    dEzz_dmz = 2 * mz
    dExz_dmz = mx
    dEyz_dmz = my

    # Compute the derivatives of second (2nd) order
    # expectations with respect to St.
    dExx_dSxx = 1
    dEyy_dSyy = 1
    dEzz_dSzz = 1
    dExy_dSxy = 1
    dExz_dSxz = 1
    dEyz_dSyz = 1

    # Compute the derivatives of third (3rd) order
    # expectations with respect to mt.
    dExxy_dmx = 2 * Exy
    dExxz_dmx = 2 * Exz
    dExyy_dmx = Eyy
    dExzz_dmx = Ezz
    dExyz_dmx = Eyz
    # ---
    dExxy_dmy = Exx
    dExyy_dmy = 2 * Exy
    dExyz_dmy = Exz
    # ---
    dExxz_dmz = Exx
    dExzz_dmz = 2 * Exz
    dExyz_dmz = Exy

    # Compute the derivatives of third (3rd) order
    # expectations with respect to St.
    dExxy_dSxx = my
    dExxz_dSxx = mz
    dExxy_dSxy = 2 * mx
    dExyy_dSxy = 2 * my
    dExyz_dSxy = mz
    dExzz_dSxz = 2 * mz
    dExyz_dSxz = my
    dExxz_dSxz = 2 * mx
    dExyy_dSyy = mx
    dExyz_dSyz = mx
    dExzz_dSzz = mx

    # Compute the derivatives of forth (4th) order expectations w.r.t. to mt.
    dExxyy_dmx = 2 * Exyy
    dExxzz_dmx = 2 * Exzz
    dExxyy_dmy = 2 * Exxy
    dExxzz_dmz = 2 * Exxz

    # Compute the derivatives of forth (4th) order expectations w.r.t. to St.
    dExxyy_dSxx = Eyy
    dExxzz_dSxx = Ezz
    dExxyy_dSxy = 4 * Exy
    dExxzz_dSxz = 4 * Exz
    dExxyy_dSyy = Exx
    dExxzz_dSzz = Exx

    # Compute the expectation for the dEsde(t)/dm(t).
    dmx1 = (
        dExx_dmx*(vS**2 + A11**2) +
        2*(dExy_dmx*(-vS**2 + vS*A11 - vS*A12 + A11*A12) + dExz_dmx*(A11 - vS)*A13 - vS*A11*dExx_dmx + b1*(vS - A11))
    )
    dmx2 = (
        dExxzz_dmx + dExx_dmx*(vR**2 + A21**2) +
        2*(dExy_dmx*(-vR + vR*A22 - A21 + A21*A22) + dExz_dmx*(vR*A23 + b2 + A21*A23) + dExyz_dmx*(1 - A22) -
           vR*dExxz_dmx + vR*A21*dExx_dmx - A21*dExxz_dmx - A23*dExzz_dmx - b2*(vR + A21))
    )
    dmx3 = (
        dExxyy_dmx + (A31**2)*dExx_dmx +
        2*(dExy_dmx*(A31*A32 - b3) + dExz_dmx*(A33 - vB)*A31 + dExyz_dmx*(A33 - vB) + A31*dExxy_dmx + A32*dExyy_dmx - A31*b3)
    )
    dmy1 = (
        dEyy_dmy*(vS**2 + A12**2) +
        2*(dExy_dmy*(-(vS**2) + vS*A11 - vS*A12 + A11*A12) + dEyz_dmy*(vS + A12)*A13 + vS*A12*dEyy_dmy - b1*(vS+A12))
    )
    dmy2 = (
        dEyy_dmy*(1 + A22**2) +
        2*(dExy_dmy*(-vR + vR*A22 - A21 + A21*A22) + dExyz_dmy*(1 - A22) - A22*dEyy_dmy + dEyz_dmy*(A22*A23 - A23) + b2*(1-A22))
    )
    dmy3 = (
        dExxyy_dmy + (A32**2)*dEyy_dmy +
        2*(dExyz_dmy*(A33 - vB) + A31*dExxy_dmy + A32*dExyy_dmy + dExy_dmy*(A31*A32 - b3) + dEyz_dmy*(A33 - vB)*A32 - A32*b3)
    )
    dmz1 = (
        (A13**2)*dEzz_dmz + 2*(dEyz_dmz*(vS+A12) + dExz_dmz*(A11-vS) - b1)*A13
    )
    dmz2 = (
        dExxzz_dmz + (A23**2)*dEzz_dmz +
        2*(dExxz_dmz*(-vR - A21) + dExz_dmz*(vR*A23 + b2 + A21*A23) + dExyz_dmz*(1 - A22) + dEyz_dmz*(A22*A23 - A23) -A23*(dExzz_dmz + b2)))
    dmz3 = (
        dEzz_dmz*(vB**2 + A33**2) +
        2*((A33 - vB)*(dExyz_dmz + dExz_dmz*A31 + dEyz_dmz*A32 - b3) - vB*A33*dEzz_dmz))

    # Gradient of the energy with respect to the marginal mean.
    dEsde_dm = 0.5 * np.array([
        [dmx1, dmx2, dmx3],
        [dmy1, dmy2, dmy3],
        [dmz1, dmz2, dmz3]
    ]).dot(diagSigI)

    # Take the diagonal elements.
    iSx, iSy, iSz = diagSigI

    # Compute the expectation for the dEsde(t)/dS(t).
    dSxx = (
        iSx*((vS - A11)**2)*dExx_dSxx +
        iSy*(dExxzz_dSxx + dExx_dSxx*((vR + A21)**2) - 2*dExxz_dSxx*(vR + A21)) +
        iSz*(dExxyy_dSxx + (A31**2)*dExx_dSxx + 2*A31*dExxy_dSxx)
    )
    dSxy = (
        iSx*2*(vS*A11 - vS**2 - vS*A12 + A11*A12)*dExy_dSxy +
        iSy*2*(dExy_dSxy*(vR*A22 - vR - A21 + A21*A22) + dExyz_dSxy*(1 - A22)) +
        iSz*(dExxyy_dSxy + 2*(dExyz_dSxy*(A33 - vB) + A31*dExxy_dSxy + A32*dExyy_dSxy + dExy_dSxy*(A31*A32 - b3)))
    )
    dSxz = (
        iSx*2*(A11 - vS)*A13*dExz_dSxz +
        iSy*(dExxzz_dSxz +
             2*(dExz_dSxz*(vR*A23 + b2 + A21*A23) + dExyz_dSxz*(1 - A22) - dExxz_dSxz*(vR + A21) - A23*dExzz_dSxz)) +
        iSz*2*(dExz_dSxz*(A33 - vB)*A31 + dExyz_dSxz*(A33 - vB))
    )
    dSyy = (
        iSx*((vS + A12)**2)*dEyy_dSyy +
        iSy*((1 - A22)**2)*dEyy_dSyy +
        iSz*(dExxyy_dSyy + (A32**2)*dEyy_dSyy + 2*A32*dExyy_dSyy)
    )
    dSyz = (
        iSx*2*(vS + A12)*A13*dEyz_dSyz +
        iSy*2*(dExyz_dSyz*(1 - A22) + dEyz_dSyz*(A22 - 1)*A23) +
        iSz*2*(dExyz_dSyz*(A33 - vB) + dEyz_dSyz*(A33 - vB)*A32)
    )
    dSzz = (
        iSx*(A13**2)*dEzz_dSzz +
        iSy*(dExxzz_dSzz + (A23**2)*dEzz_dSzz - 2*A23*dExzz_dSzz) +
        iSz*((vB - A33)**2)*dEzz_dSzz
    )

    # Gradient of the energy with respect to the marginal covariance.
    dEsde_dS = 0.5*np.array([
        [dSxx, dSxy, dSxz],
        [dSxy, dSyy, dSyz],
        [dSxz, dSyz, dSzz]
    ])

    return Efg, dEsde_dm, dEsde_dS


def Efg_drift_theta(At, bt, mt, St, sDyn):
    """
    EFG_DRIFT_THETA

    Description:
    Returns expectation : <(f-g)'*(df/dtheta)>.
    It is used when estimating the drift parameters.

    [Input parameters]:
    At  : variational linear parameter. (3 x 3).
    bt  : variational offset parameter. (1 x 3).
    mt  : marginal mean (1 x 3).
    St  : marginal covariance  (3 x 3).

    [Output parameters]:
    Gpar : gradient w.r.t. THETA (1 x 3).

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
    Exxz = Sxx*mz + 2*Sxz*mx + (mx**2)*mz
    Exyz = Sxy*mz + Sxz*my + Syz*mx + mx*my*mz

    # Compute the expectation.
    V1 = Eyy*(vS + A12) + Exx*(vS - A11) + Exy*(A11 - 2*vS - A12) + A13*(Eyz - Exz) + b1*(mx - my)
    V2 = vR*Exx - Exy - Exxz + A21*Exx + A22*Exy + A23*Exz - b2*mx
    V3 = -Exyz + vB*Ezz - A31*Exz - A32*Eyz - A33*Ezz + b3*mz

    return np.array([V1, V2, V3])

