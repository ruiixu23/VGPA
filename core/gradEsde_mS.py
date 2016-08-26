# Imports:
import numpy as np

# Public functions:
__all__ = ['gradEsde_mS']

# Listing: xx
def gradEsde_mS(x, F, mt, St, At, bt, diagSigI, sDyn=None):
    """
        GRADIENT of ESDE w.r.t MEAN and STD
    
    Description:
    Returns the gradient of the -SDE- energy function with respect
    to the marginal means and variances . This method is used when
    the analytic expressions for the gradients are difficult to be
    computed, hence we use approximations such as the unscented
    transformation.
    
    [Input parameters]:
    x        : input state samples (K x D).
    F        : drift function.
    mt       : marginal mean at time 't' (1 x D).
    St       : marginal covar. at time 't' (D x D).
    At       : linear parameter (D x D).
    bt       : offset parameter (1 x D).
    diagSigI : diagonal elements of inverse system noise (1 x D).
    sDyn     : optional additional parameters for the model function.
    
    [Output parameters]:
    dmS      : gradient w.r.t. to 'mt' and 'St' [K x D*(D+1)].
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Updated: November 2015.
    """
    
    # Get the dimensions of the input array.
    K, D = x.shape
    
    # Preallocate array: [K x D^2]
    dSt = np.zeros((K,D*D), dtype='float64')
    
    # Compute auxiliary quantity:
    xMat = (F(x,sDyn) + x.dot(At.T) - np.tile(bt,(x.shape[0],1)))**2
    v = diagSigI.dot(xMat.T) # [1 x K]
    
    # Gradient w.r.t. 'mt': [K x D]
    dmt = np.linalg.solve(St,(np.tile(v,(D,1))*x.T)).T
    
    # Inverse of marginal covariance.
    Sti = np.linalg.inv(St)
    
    # Calculate the gradients w.r.t. 'St'.
    for k in range(K):
        # Take the values at sample 'k'.
        zt = np.reshape(x[k,:] - mt, (1,D))
        # Square matrix.
        Zk = zt.T.dot(zt)
        # Gradient w.r.t. 'St'.
        dSt[k,:] = v[k] * np.linalg.solve(St,Zk).dot(Sti).ravel()
    # ...
    
    # Scale the results.
    dmt = 0.5*dmt
    dSt = 0.5*dSt
    
    # Group the gradients together and exit:
    # The dimensions are: [K x D]+[K x D^2]
    return np.concatenate((dmt, dSt), axis=1)

# End-Of-File