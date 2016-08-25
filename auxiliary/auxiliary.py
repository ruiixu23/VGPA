# Imports:
import numpy as np
from numpy.linalg import inv, LinAlgError
from numpy.linalg import cholesky as chol

# Public functions:
__all__ = ['chol_inv', 'log_det', 'finiteDiff', 'ut_approx', 'mytrapz', 'safelog']

# Listing: 01
def chol_inv(A):
    """
        CHOLINV
    
    Description:
    Returns the inverse of a matrix by using Cholesky decomposition.
    
    Input:
    A  : input array (D x D).
    
    Output:
    Ai : Inverse of A (D x D).
    Ri : Inverted Cholesky factor.
    
    References:
        (N/A)
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Updated: November 2015.
    """
    
    # Check if input is empty.
    if (A is None):
        print("Input matrix is None!")
        return None, None
    
    # Check if input is scalar.
    if np.isscalar(A):
        return 1.0/A, 1.0/np.sqrt(A)
    else:
        # If the input is vector.
        if (len(A.shape) == 1):
            # Transform it to diagonal matrix
            A = np.diag(A)
    
    # Try Cholesky decomposition.
    Ri = inv(chol(A))
    Ai = Ri.dot(Ri.T)
    
    # --->
    return Ai, Ri

# Listing: 02
def log_det(A):
    """
        LOG(DET)
    
    Description:
    Returns the log(det(A)), but more stable and accurate.
    
    Input:
    A : input array (D x D).
    
    Output:
    X : log(det(A)) (D x D).
    
    References:
        (N/A)
    
    Copyright (c) Michail D. Vrettas - November 2015.
    
    Last Updated: November 2015.
    """
    # Check if input is empty.
    if (A is None):
        print("Input matrix is None!")
        return None
    
    # Check if input is scalar.
    if np.isscalar(A):
        return log(A)
    else:
        # If the input is vector.
        if (len(A.shape) == 1):
            # Transform it to diagonal matrix
            A = np.diag(A)
    
    # --->
    return 2*np.sum(np.log(chol(A).diagonal()))

# Listing: 03
def finiteDiff(fun, x, mParam, h = 1.0e-6):
    """
        FINITEDIFF
    
    Description:
    Calculates the approximate derivative of function "fun"
    on a parameter vector "x". A central difference formula
    with step size "h" is used, and the result is returned
    in vector "gradN".
    
    Input  :
    fun    : the objective function that we want to check.
    x      : the point where we want to check the gradient.
    mParam : additional function parameters.
    h      : stepzise for CD formula (optional).
    
    Output :
    gradN  : gradient calculated numerically.
    
    References:
        (N/A)
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
     
    Last Update: November 2015
    """    
    
    # Number of input parameters (Dimensionality of input vector).
    D = x.size
    
    # Make input a column-vector.
    x = x.reshape(D,1)
    
    # Preallocate array.
    gradN = np.zeros((D,1))
    
    # Unit vector.
    e = np.zeros((D,1))
    
    # Initial message.
    print(' Checking numerically the gradient ...')
    
    # Check all 'D' directions (coordinates of x).
    for i in range(D):
        # Switch ON i-th direction.
        e[i] = 1.0
        
        # Move a small way in the i-th direction of x.
        fplus  = fun(x+h*e, mParam)
        fminus = fun(x-h*e, mParam)
        
        # Use central difference formula for approximation.
        gradN[i] = 0.5*(fplus - fminus)/h
        
        # Switch OFF i-th direction
        e[i] = 0.0;
        
        # Display progress so far ...
        if (i%100 == 0):
            print('Done: {0:.3}%'.format(i/D*100))
    # --->
    return gradN

# Listing: 04
def ut_approx(f, xbar, Pxx, *args):
    """
        UNSCENTED TRANSFORMATION
    
    Description:
    This method computes the approximate values for the mean and
    the covariance of a multivariate random variable. To achieve
    that, the "Unscented Transformation" (UT) is used.
    
    Input :
    f     : function of the nonlinear transformation of the state vector.
    xbar  : current mean of the state vector        (1 x D).
    Pxx   : current covarriance of the state vector (D x D).
    *args : additional parameter for the "f" function.
    
    Output:
    ybar  : estimated mean after nonlinear transformation       (1 x K).
    Pyy   : estimated covariance after nonlinear transformation (K x K).
    
    Reference:
    @INPROCEEDINGS{Julier99thescaled,
        author = {Simon J. Julier},
        title = {The Scaled Unscented Transformation},
        booktitle = {},
        year = {1999},
        pages = {4555--4559}
    }
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Update: November 2015.
    """
    
    # Get the dimensions of the state vector.
    if (len(xbar.shape) == 1):
        D = xbar.shape[0]
    else:
        D = xbar.shape[1]
    
    # Total number of sigma points.
    M = (2*D + 1)
    
    # Scaling factor.
    k = 1.05*D
    
    # Use Cholesky to get the lower triangular matrix.
    try:
        sPxx = chol((D+k)*Pxx).T
    except LinAlgError:
        sPxx = chol(Pxx*eye(D)).T
    
    # Replicate the array.
    xMat = np.tile(xbar,(D,1))
    
    # Put all sigma points together.
    chi = np.concatenate((xbar[np.newaxis,:], (xMat+sPxx), (xMat-sPxx)))
    
    # Compute the weights.
    wList = [k/(D+k)]; wList.extend([1.0/(2.0*(D+k))]*(M-1))
    weights = np.reshape(np.array(wList), (1,M))
    
    # Propagate the new points through the nonlinear transformation.
    Y = f(chi, *args)
    
    # Compute the new approximate mean.
    ybar = weights.dot(Y)
    
    # Compute the approximate covariance.
    wM = np.eye(M) - np.tile(weights,(M,1))
    Q = wM.dot(np.diag(weights.ravel())).dot(wM.T)
    
    # Compute the new approximate covariance.
    Pyy = Y.T.dot(Q).dot(Y)
    
    # --->
    return ybar, Pyy

# Listing: 05
def mytrapz(fxt, dt, idx=None):
    """
        TRAPZ (NUMERICAL INTEGRATION)
    
    Description:
    This method computes the numerical integral of the discrete function values
    'fx', with space increment dt, using the composite trapezoidal rule.
    This code applies the function: numpy.trapz(y, x=None, dx=1.0, axis=-1)
    between the times the observations occur 'idx'. This is because the function
    fx is very rough (it jumps at observation times), so by computing the
    integral incrementally we achieve better numerical results. If no 'idx'
    is given, then we call directly trapz().
    
    Input:
    fx   : function values (discrete).
    dt   : discretization step (time-wise)
    idx  : observation times (indexes) - optional
    
    Output:
    trapz : Definite integral as approximated by trapezoidal rule.
    
    See Also
    --------
    numpy.trapz
    
    References:
        (N/A)
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Update: November 2015.
    """
    
    # Check if there are observation times indexes.
    if idx is None:
        return np.trapz(fxt, dx=dt, axis=0)
    
    # Initialization.
    v = 0.0
    
    # First index.
    f = 0
    
    # Compute the integral partially.
    for n in range(len(idx)):
        # Get the index of the observation.
        l = idx[n]
        # Compute the integral incrementally.
        v += np.trapz(fxt[f:l], dx=dt, axis=0)
        # Set the next first index.
        f = l
    
    # Return the total integral.    
    return v

# Listing: 06
def safelog(x = None):
    '''
        SAFE LOG
    
    Description:
        This (helper) function prevents the computation of very small or very  large
        values of logarithms that would lead to -/+ inf, by setting predefined LOWER
        and UPPER bounds. The bounds are set as follows:
        
            - LOWER = 1.0E-300
            - UPPER = 1.0E+300
            
        It is assumed that the input values lie within this range.
    
    Example:
        >> numpy.log(1.0E-350)
        >> -inf
        >>
        >> safelog(1.0E-350)
        >> -690.77552789821368
    
    Input:
        x : input array (N x M).
        
    Output:
        x : the log(x) after the values of x have been filtered (N x M).
    
    References:
        (N/A)
    
    Copyright (c) Michail D. Vrettas, PhD - March 2015.
    
    Last Update: March 2015.
    '''
    
    # Prevent empty input.
    if(x == None):
        print(" [safelog] in debug mode ... exiting:", end="")
        return None
    
    # Define LOWER and UPPER bounds.
    _LWR_Bound = 1.0E-300
    _UPR_Bound = 1.0E+300
    
    # Make sure input is an array.
    x = np.asarray(x)
    
    # Check for scalar.
    if (x.ndim == 0):
        if (x <_LWR_Bound):
            x =_LWR_Bound
        elif (x >_UPR_Bound):
            x =_UPR_Bound
        #_end_if
    else:
        # Check Lower/Upper bounds.
        x[x <_LWR_Bound] =_LWR_Bound
        x[x >_UPR_Bound] =_UPR_Bound
    
    # Return the log() of the filtered input.
    return np.log(x)

# End-Of-File