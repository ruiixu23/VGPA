# Imports:
import numpy as np
from copy import deepcopy
from core.varFreeEnergy import varFreeEnergy

# Public functions:
__all__ = ['gradL_Ab']

# Listing 01:
def gradL_Ab(Ab, mParam, evalF=False):
    """
       GRADIENT W.R.T AB

    Description:
    Returns the gradient of the Lagrangian w.r.t. the variational
    parameters A (linear) and b (offset).

    [Input parameters]:
    Ab     : variational linear+offset parameters (N*D*(D + 1)).
    mParam : sde structure includes:
            m   : means (N x D).
            S   : covariances (N x D x D).
            lam : lagrange multipliers for m-constraint (N x D).
            Psi : lagrange multipliers for S-constraint (N x D x D).
            Ef  : average drift (N x D).
            Edf : average differentiated drift (N x D).

    [Output parameters]:
    gradL  : grouped gradient of the Lagrangian w.r.t. the variational
             linear parameters 'A' (N x D x D) and the variational offset
             parameters 'b' (N x D).
    """

    # Check whether we have to evaluate the cost function.
    if evalF:
        # Make a local copy of the parameters.
        mParam = deepcopy(mParam)
        # Run again the cost function to enforce the consistency
        # of the variational parameters with the rest of the terms.
        varFreeEnergy(Ab, mParam)

    # Call the right version of the gradients.
    if (mParam['sde_struct']['D'] == 1):
        return gradL_Ab_1D(Ab, mParam)
    else:
        return gradL_Ab_nD(Ab, mParam)
    # ...

# Listing 02:
def gradL_Ab_1D(Ab, mParam):
    """
       GRAD_AB_1D : Works with 1D systems.
    """

    # Inverse system noise.
    SigInv = 1.0/mParam['sde_struct']['Sig']

    # Extract model parameters.
    N = mParam['sde_struct']['N']

    # Time discretization step.
    dt = mParam['sde_struct']['dt']

    # Unpack data.
    A = Ab[:N]
    b = Ab[N:]

    # Posterior moments: m(t), S(t).
    m = mParam['mt']
    S = mParam['St']

    # Lagrange multipliers: lam(t), Psi(t).
    lam = mParam['lamt']
    Psi = mParam['Psit']

    # Expectations.
    Efx = mParam['Efx']
    Edf = mParam['Edf']

    # Preallocate the return arrays.
    gLA = np.zeros((N,1), dtype='float64')
    gLb = np.zeros((N,1), dtype='float64')

    # Main loop.
    for t in range(N):
        # Get the values at time 't'.
        At = A[t]
        St = S[t]
        mt = m[t]
        lamt = lam[t]

        # Gradient of Esde w.r.t. 'b' -Eq(29)-
        dEsde_dbt = SigInv*(-Efx[t] - At*mt + b[t])

        # Gradient of Esde w.r.t. 'A' -Eq(28)-
        dEsde_dAt = SigInv*(Edf[t] + At)*St - dEsde_dbt*mt

        # Gradient of Lagranian w.r.t. 'A' -Eq(12)-
        gLA[t] = dEsde_dAt - lamt*mt - 2.0*Psi[t]*St

        # Gradient of Lagranian w.r.t. 'b' -Eq(13)-
        gLb[t] = dEsde_dbt + lamt
    # ...

    # Scale the results with the time increment.
    gLA = dt*gLA
    gLb = dt*gLb

    # Group the gradients together and exit.
    return np.concatenate((gLA,gLb))

# Listing 03:
def gradL_Ab_nD(Ab, mParam):
    """
       GRAD_AB_nD: Works with n-D systems.
    """

    # System noise.
    Sigma = mParam['sde_struct']['Sig']

    # Extract model parameters.
    N = mParam['sde_struct']['N']
    D = mParam['sde_struct']['D']

    # Time discretization step.
    dt = mParam['sde_struct']['dt']

    # Total number of linear parameters 'A'.
    K = N*D*D

    # Unpack data.
    A = Ab[:K].reshape(N,D,D)
    b = Ab[K:].reshape(N,D)

    # The Inverse of System Noise covariance.
    # NB: Ideally we shouldn't directly invert the matrix,
    # but since the Sigma is diagonal this operation is OK.
    SigInv = np.linalg.inv(Sigma)

    # Posterior moments: m(t), S(t).
    m = mParam['mt']
    S = mParam['St']

    # Lagrange multipliers: lam(t), Psi(t).
    lam = mParam['lamt']
    Psi = mParam['Psit']

    # Expectations.
    Efx = mParam['Efx']
    Edf = mParam['Edf']

    # Preallocate the return arrays.
    gLA = np.zeros((N,D,D),dtype='float64')
    gLb = np.zeros((N,D),  dtype='float64')

    # Main loop.
    for t in range(N):
        # Get the values at time 't'.
        At = A[t,:,:]
        St = S[t,:,:]
        mt = m[t,:]
        lamt = lam[t,:]

        # Gradient of Esde w.r.t. 'b' -Eq(29)-
        dEsde_dbt = (-Efx[t,:] - mt.dot(At.T) + b[t,:]).dot(SigInv.T)

        # Gradient of Esde w.r.t. 'A' -Eq(28)-
        dEsde_dAt = SigInv.dot(Edf[t,:,:] + At).dot(St) - dEsde_dbt.T.dot(mt)

        # Gradient of Lagranian w.r.t. 'A' -Eq(12)-
        gLA[t,:,:] = dEsde_dAt - lamt.T.dot(mt) - 2.0*Psi[t,:,:].dot(St)

        # Gradient of Lagranian w.r.t. 'b' -Eq(13)-
        gLb[t,:] = dEsde_dbt + lamt
    # ...

    # Scale the results with the time increment.
    gLA = dt*gLA
    gLb = dt*gLb

    # Group the gradients together and exit.
    return np.concatenate((gLA.ravel()[:,np.newaxis],\
                           gLb.ravel()[:,np.newaxis]))

# End-Of-File
