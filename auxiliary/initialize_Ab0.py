# Imports:
import numpy as np
from scipy import interpolate

# Public functions:
__all__ = ['initialize_Ab0']

# Listing 01:
def initialize_Ab0(S0, sde_struct):
    """
        INITIALIZATION OF THE VARIATIONAL PARAMETERS

    [Description]
    This function initilizes the variational parameters A and b. This is
    done with a simple interpolation technique. In the case where the dim
    is more than one the interpolation happens on each dimension separately.
    
    [Input]
    S0         : marginal variance at t=0 (D x D).
    sde_struct : data structure (dictionary) that holds model parameters.
    
    [Output]
    [A0, b0] : the output is concatenated in one single vector (K x 1).
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Updated: August 2016.
    """
    
    # Extract parameters.
    Sigma = sde_struct['Sig']
    
    # Number of discrete points.
    N = sde_struct['N']
    
    # Dimensionality of the state vector.
    D = sde_struct['D']
    
    # Time window of inference.
    Tw = sde_struct['Tw']
    
    # Observation times (indexes).
    obsX = sde_struct['obsX']
    
    # Observation values.
    obsY = sde_struct['obsY']
    
    # Replicate the first and last observations.
    obsZ = np.column_stack((obsY[0], obsY.T, obsY[-1])).T
    
    # Time discretization.
    dt = sde_struct['dt']
    
    # Replicate the first and last time poitns.
    Tx = [Tw[0]]
    Tx.extend(Tw[obsX])
    Tx.extend([Tw[-1]])
    
    # Initialize according to the system's dimensionality.
    if (D == 1):
        # Linear variational parameters.
        A0 = 0.5*(Sigma/S0)*np.ones((N,1))
        
        # Build a univariate extrapolator (with cubic splines).
        fb0 = interpolate.UnivariateSpline(Tx, obsZ[:,0].T, k=3)
        
        # Generate the offset parameters on the whole time window.
        b0 = fb0(Tw)
    else:
        # Preallocate mt0.
        mt0 = np.zeros((N,D))
        
        # Build a spline extrapolator for each dimension separately.
        for i in range(D):
            fbi = interpolate.UnivariateSpline(Tx, obsZ[:,i].T, k=3)
            mt0[:,i] = fbi(Tw)
        # ...
        
        # Preallocate variational parameters.
        A0 = np.zeros((N,D,D), dtype='float64')
        b0 = np.zeros((N,D),   dtype='float64')
        
        # Compute the discrete differences (approximation of Dm(t)/Dt).
        dmt0 = np.diff(mt0,axis=0)/dt
        
        # Constract A0(t) and b0(t) assuming A0(t) and S(t) are diagonal.
        for k in range(N-1):
            A0[k,:,:] = 0.5*np.diag(Sigma.diagonal()/S0.diagonal())
            b0[k,:] = dmt0[k,:] + A0[k,:,:].diagonal()*mt0[k,:]
        
        # At the last points (t=tf) assume the gradien Dmt0 is zero.
        A0[-1,:,:] = 0.5*np.diag(Sigma.diagonal()/S0.diagonal())
        b0[-1,:] = A0[-1,:,:].diagonal()*mt0[-1,:]
    
    # Concatenate the results into one (big) array before exit.
    return np.concatenate((A0.ravel()[:,np.newaxis],\
                           b0.ravel()[:,np.newaxis]))

# End-Of-File.