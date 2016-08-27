"""
    Forward ODE integrators.

    This module implements a set of forward integration methods for the
    computation of the marginal posterior moments m(t) and S(t) of the
    variational algorithms. To make it easier and more broadly applicable
    the algorithm detects the dimensions of the state vector (D) and calls
    either the 1D or nD version of the selected algorithm. This is because
    there are signficant performance gains when writing the code in 1D
    rather than nD. Default values exist in the integration method ('Euler')
    and time step ('dt = 0.01') to prevent common errors.

	List:
        The module include the following methods in both 1D and nD versions
        (a detailed description is given at the individuals methods):

        1) Euler       (1st Order)
        2) Heun        (2nd Order)
        3) Runge-Kutta (2nd Order)
        4) Runge-Kutta (4th Order)
    
	Author:
		Michail D. Vrettas, PhD

	Contact:
		For any bug or suggestions, please contact me at:
		<vrettasm@gmail.com>
	
	Webpage:
		The code can be downloaded from:
        <http://vrettasm.weebly.com/software.html>

	Copyright (c) Michail D. Vrettas, PhD - November 2015.
"""

# Import ONLY the functions you need.
from numpy import zeros

# Public functions:
__all__ = ['solver']

# Define (lambda) functions for the means and covariances for the nD versions.
# These are very helpful in the Runge-Kutta implementations!
fun_mt = lambda mt, At, bt: (-mt.dot(At.T) + bt)
fun_Ct = lambda St, At, Sn: (-At.dot(St) - St.dot(At.T) + Sn)

# Main (callable) method.
def solver(A, b, m0, S0, Sigma, dt=0.01, method='EULER'):

    # Check whether the system is 1D or nD
    if (len(A.shape) == 1):
        nD = 1
    else:
        nD = A.shape[1]

    # Switch to the requested algorithm.
    if (method.upper()=='EULER'):
        # In the 1-D case call a faster version.
        if (nD == 1):
            return euler_1D(A, b, m0, S0, Sigma, dt)
        else:
            return euler_nD(A, b, m0, S0, Sigma, dt)
    elif (method.upper()=='HEUN'):
        # In the 1-D case call a faster version.
        if (nD == 1):
            return heun_1D(A, b, m0, S0, Sigma, dt)
        else:
            return heun_nD(A, b, m0, S0, Sigma, dt)
    elif (method.upper()=='RK2'):
        # In the 1-D case call a faster version.
        if (nD == 1):
            return rk2_1D(A, b, m0, S0, Sigma, dt)
        else:
            return rk2_nD(A, b, m0, S0, Sigma, dt)
    elif (method.upper()=='RK4'):
        # In the 1-D case call a faster version.
        if (nD == 1):
            return rk4_1D(A, b, m0, S0, Sigma, dt)
        else:
            return rk4_nD(A, b, m0, S0, Sigma, dt)
    else:
        raise ValueError('Unknown method of integration')


# Function: Euler
# It can be called directly, but it is preferred to call it through: solver()
def euler_nD(A, b, m0, S0, Sigma, dt):
    """ Euler integration method.

        This function implements a simple Euler integration method.

        Args:
          A (array N x D x D): Linear variational parameters
          b     (array N x D): Offset variational parameters
          m0    (array 1 x D): Initial conditions (means)
          S0    (array D x D): Initial conditions (covar)
          Sigma (array D x D): System noise covariance
          dt          (float): time (integration) step

        Returns:
          m: posterior means values (N x D)
          S: posterior covar values (N x D)

          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """

    # Get the dimensions.
    # D: is the number of the system states
    # N: is the number of discrete time points
    (N, D) = b.shape

    # Preallocate the return arrays.
    m = zeros((N,D),  dtype='float64')
    S = zeros((N,D,D),dtype='float64')

    # Initialize the first moments.
    m[0,:] = m0; S[0,:,:] = S0

    # Run through all time points.
    for t in range(N-1):
        # Get the values at time 't'.
        At = A[t,:,:]; bt = b[t,:]
        St = S[t,:,:]; mt = m[t,:]

        # -Eq(09)- NEW "mean" point.
        m[t+1,:] = mt + fun_mt(mt, At, bt)*dt

        # -Eq(10)- NEW "covariance" point.
        S[t+1,:,:] = St + fun_Ct(St, At, Sigma)*dt
    #_end_for_
    return (m, S)

# Function: Euler-1D
def euler_1D(A, b, m0, S0, Sigma, dt):
    """ Euler integration method.

        Same as euler_nD but for 1D models only.

        Args:
          A     (array N x 1): Linear variational parameters
          b     (array N x 1): Offset variational parameters
          m0    (array 1 x 1): Initial conditions (means)
          S0    (array 1 x 1): Initial conditions (covar)
          Sigma (array 1 x 1): System noise variance
          dt          (float): time (integration) step

        Returns:
          m: posterior means values (N x 1)
          S: posterior covar values (N x 1)

          The return vectors are 'None' if an error has occurred
          while integrating the ODEs.
    """

    # Get 'N': the number of discrete time points
    N = b.shape[0]

    # Preallocate the return arrays.
    m = zeros((N,1), dtype='float64')
    S = zeros((N,1), dtype='float64')

    # Initialize the first moments.
    m[0] = m0; S[0] = S0

    # Run through all time points.
    for t in range(N-1):
        # Get the values at time 't'.
        At = A[t]; bt = b[t]
        St = S[t]; mt = m[t]

        # -Eq(09)- NEW "mean" point.
        m[t+1] = mt + (-At*mt + bt)*dt

        # -Eq(10)- NEW "covariance" point.
        S[t+1] = St + (-2*At*St + Sigma)*dt
    #_end_for_
    return (m, S)

# Function: Heun
def heun_nD(A, b, m0, S0, Sigma, dt):
    """ Heun integration method.

        This function implements a simple Heun integration method.

        Args:
          A (array N x D x D): Linear variational parameters
          b     (array N x D): Offset variational parameters
          m0    (array 1 x D): Initial conditions (means)
          S0    (array D x D): Initial conditions (covar)
          Sigma (array D x D): System noise covariance
          dt          (float): time (integration) step

        Returns:
          m: posterior means values (N x D)
          S: posterior covar values (N x D)

          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """

    # Get the dimensions.
    # D: is the number of the system states
    # N: is the number of discrete time points
    (N, D) = b.shape

    # Preallocate the return arrays.
    m = zeros((N,D),  dtype='float64')
    S = zeros((N,D,D),dtype='float64')

    # Half stepsize.
    h = 0.5*dt

    # Initialize the first moments.
    m[0,:] = m0; S[0,:,:] = S0

    # Run through all time points.
    for t in range(N-1):
        # Get the values at time 't'.
        At = A[t,:,:]; bt = b[t,:]
        St = S[t,:,:]; mt = m[t,:]

        # Get the value at time 't+1'.
        Ak = A[t+1,:,:]; bk = b[t+1,:]

        # -Eq(09)- Prediction step:
        ftp = fun_mt(mt, At, bt)

        # Correction step:
        ftc = fun_mt((mt + ftp*dt), Ak, bk)

        # NEW "mean" point.
        m[t+1,:] = mt + h*(ftp + ftc)

        # -Eq(10)- Prediction step:
        ftp = fun_Ct(St, At, Sigma)

        # Correction step:
        ftc = fun_Ct((St + ftp*dt), Ak, Sigma)

        # NEW "variance" point.
        S[t+1,:,:] = St + h*(ftp + ftc)
    #_end_for_
    return (m, S)

# Function: Heun-1D
def heun_1D(A, b, m0, S0, Sigma, dt):
    """ Heun integration method.

        Same as heun_nD but for 1D models only.

        Args:
          A     (array N x 1): Linear variational parameters
          b     (array N x 1): Offset variational parameters
          m0    (array 1 x 1): Initial conditions (means)
          S0    (array 1 x 1): Initial conditions (variance)
          Sigma (array 1 x 1): System noise variance
          dt          (float): time (integration) step

        Returns:
          m: posterior means values (N x 1)
          S: posterior covar values (N x 1)

          The return vectors are 'None' if an error has occurred
          while integrating the ODEs.
    """

    # Get 'N': the number of discrete time points
    N = b.shape[0]

    # Preallocate the return arrays.
    m = zeros((N,1), dtype='float64')
    S = zeros((N,1), dtype='float64')

    # Half stepsize.
    h = 0.5*dt

    # Initialize the first moments.
    m[0] = m0; S[0] = S0

    # Run through all time points.
    for t in range(N-1):
        # Get the values at time 't'.
        At = A[t]; bt = b[t]
        St = S[t]; mt = m[t]

        # Get the value at time 't+1'.
        Ak = A[t+1]; bk = b[t+1]

        # -Eq(09)- Prediction step:
        ftp = -At*mt + bt

        # Correction step:
        ftc = -Ak*(mt + ftp*dt) + bk

        # NEW "mean" point.
        m[t+1] = mt + h*(ftp + ftc)

        # -Eq(10)- Prediction step:
        ftp = -2*At*St + Sigma

        # Correction step:
        ftc = -2*Ak*(St + ftp*dt) + Sigma

        # NEW "variance" point.
        S[t+1] = St + h*(ftp + ftc)
    #_end_for_
    return (m, S)

# Function: Runge-Kutta 2
def rk2_nD(A, b, m0, S0, Sigma, dt):
    """ Runge-Kutta 2nd order integration method.

        This function implements a RK2 integration method.

        Args:
          A (array N x D x D): Linear variational parameters
          b     (array N x D): Offset variational parameters
          m0    (array 1 x D): Initial conditions (means)
          S0    (array D x D): Initial conditions (covar)
          Sigma (array D x D): System noise covariance
          dt          (float): time (integration) step

        Returns:
          m: posterior means values (N x D)
          S: posterior covar values (N x D)

          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """

    # Get the dimensions.
    # D: is the number of the system states
    # N: is the number of discrete time points
    (N, D) = b.shape

    # Preallocate the return vector/matrix.
    m = zeros((N,D),  dtype='float64')
    S = zeros((N,D,D),dtype='float64')

    # Half stepsize.
    h = 0.5*dt

    # Initialize the first moments.
    m[0,:] = m0; S[0,:,:] = S0

    # Run through all time points.
    # Implemented in one step for faster execution.
    for t in range(N-1):
        # Get the values at time 't'.
        At = A[t,:,:]; bt = b[t,:]
        St = S[t,:,:]; mt = m[t,:]

        # Get the midpoints at time 't + 0.5*dt'.
        Ak = 0.5*(At + A[t+1,:,:])
        bk = 0.5*(bt + b[t+1,:])

        # -Eq(09)- NEW "mean" point.
        m[t+1,:] = mt + fun_mt((mt + h*fun_mt(mt, At, bt)), Ak, bk)*dt

        # -Eq(10)- NEW "variance" point.
        S[t+1,:,:] = St + fun_Ct((St + h*fun_Ct(St, At, Sigma)), Ak, Sigma)*dt
    #_end_for_
    return (m, S)

# Function: Runge-Kutta2 1D
def rk2_1D(A, b, m0, S0, Sigma, dt):
    """ Runge-Kutta 2nd order integration method

        Same as rk2_nD but for 1D models only.

        Args:
          A     (array N x 1): Linear variational parameters
          b     (array N x 1): Offset variational parameters
          m0    (array 1 x 1): Initial conditions (means)
          S0    (array 1 x 1): Initial conditions (covar)
          Sigma (array 1 x 1): System noise variance
          dt          (float): time (integration) step

        Returns:
          m: posterior means values (N x 1)
          S: posterior covar values (N x 1)

          The return vectors are 'None' if an error has occurred
          while integrating the ODEs.
    """

    # Get the dimensions.
    # N: is the number of discrete time points
    N = b.shape[0]

    # Preallocate the return vector/matrix.
    m = zeros((N,1), dtype='float64')
    S = zeros((N,1), dtype='float64')

    # Half stepsize.
    h = 0.5*dt

    # Define locally (lambda) functions for the means and variances.
    Fm = lambda mt, At, bt:  (-At*mt + bt)
    Fs = lambda St, At, Sig: (-2*At*St + Sig)

    # Initialize the first moments.
    m[0] = m0; S[0] = S0

    # Run through all time points.
    # Implemented in one step for faster execution.
    for t in range(N-1):
        # Get the values at time 't'.
        At = A[t]; bt = b[t]
        St = S[t]; mt = m[t]

        # Get the midpoints at time 't + 0.5*dt'.
        Ak = 0.5*(At + A[t+1])
        bk = 0.5*(bt + b[t+1])

        # -Eq(09)- NEW "mean" point.
        m[t+1] = mt + Fm((mt + h*Fm(mt, At, bt)), Ak, bk)*dt

        # -Eq(10)- NEW "variance" point.
        S[t+1] = St + Fs((St + h*Fs(St, At, Sigma)), Ak, Sigma)*dt
    #_end_for_
    return (m, S)

# Function: Runge-Kutta 4
def rk4_nD(A, b, m0, S0, Sigma, dt):
    """ Runge-Kutta 4th order integration method.

        This function implements a RK4 integration method.

        Args:
          A (array N x D x D): Linear variational parameters
          b     (array N x D): Offset variational parameters
          m0    (array 1 x D): Initial conditions (means)
          S0    (array D x D): Initial conditions (covar)
          Sigma (array D x D): System noise covariance
          dt          (float): time (integration) step

        Returns:
          m: posterior means values (N x D)
          S: posterior covar values (N x D)

          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """

    # Get the dimensions.
    # D: is the number of the system states
    # N: is the number of discrete time points
    (N, D) = b.shape

    # Preallocate the return vector/matrix.
    m = zeros((N,D),  dtype='float64')
    S = zeros((N,D,D),dtype='float64')

    # Initialize the first moments.
    m[0,:] = m0; S[0,:,:] = S0

    # Runge-Kutta 4 (4th-Order).
    for t in range(N-1):
        # Get the values at time 't'.
        At = A[t,:,:]; bt = b[t,:]
        St = S[t,:,:]; mt = m[t,:]

        # Get the values at time 't+1'.
        Ap = A[t+1,:,:]; bp = b[t+1,:]

        # Get the midpoints at time 't + 0.5*dt'.
        Ak = 0.5*(At + Ap)
        bk = 0.5*(bt + bp)

        # Intermediate steps.
        K1 = fun_mt(mt, At, bt)*dt
        K2 = fun_mt((mt + 0.5*K1), Ak, bk)*dt
        K3 = fun_mt((mt + 0.5*K2), Ak, bk)*dt
        K4 = fun_mt((mt + K3), Ap, bp)*dt

        # NEW "mean" point.
        m[t+1,:] = mt + (K1 + 2.0*(K2 + K3) + K4)/6.0

        # Intermediate steps.
        L1 = fun_Ct(St, At, Sigma)*dt
        L2 = fun_Ct((St + 0.5*L1), Ak, Sigma)*dt
        L3 = fun_Ct((St + 0.5*L2), Ak, Sigma)*dt
        L4 = fun_Ct((St + L3), Ap, Sigma)*dt

        # NEW "variance" point
        S[t+1,:,:] = St + (L1 + 2.0*(L2 + L3) + L4)/6.0
    #_end_for_
    return (m, S)

# Function: Runge-Kutta 4
def rk4_1D(A, b, m0, S0, Sigma, dt):
    """ Runge-Kutta 4th order integration method

        Same as rk4_nD but for 1D models only.

        Args:
          A     (array N x 1): Linear variational parameters
          b     (array N x 1): Offset variational parameters
          m0    (array 1 x 1): Initial conditions (means)
          S0    (array 1 x 1): Initial conditions (covar)
          Sigma (array 1 x 1): System noise variance
          dt          (float): time (integration) step

        Returns:
          m: posterior means values (N x 1)
          S: posterior covar values (N x 1)

          The return vectors are 'None' if an error has occurred
          while integrating the ODEs.
    """

    # Get 'N': the number of discrete time points
    N = b.shape[0]

    # Preallocate the return arrays.
    m = zeros((N,1), dtype='float64')
    S = zeros((N,1), dtype='float64')

    # Initialize the first moments.
    m[0] = m0; S[0] = S0

    # Define locally (lambda) functions for the means and variances.
    Fm = lambda mt, At, bt:  (-At*mt + bt)
    Fs = lambda St, At, Sig: (-2*At*St + Sig)

    # Runge-Kutta 4 (4th-Order).
    for t in range(N-1):
        # Get the values at time 't'.
        At = A[t]; bt = b[t]
        St = S[t]; mt = m[t]

        # Get the values at time 't+1'.
        Ap = A[t+1]; bp = b[t+1]

        # Get the midpoints at time 't + 0.5*dt'.
        Ak = 0.5*(At + Ap)
        bk = 0.5*(bt + bp)

        # Intermediate steps.
        K1 = Fm(mt, At, bt)*dt
        K2 = Fm((mt + 0.5*K1), Ak, bk)*dt
        K3 = Fm((mt + 0.5*K2), Ak, bk)*dt
        K4 = Fm((mt + K3), Ap, bp)*dt

        # NEW "mean" point.
        m[t+1] = mt + (K1 + 2.0*(K2 + K3) + K4)/6.0

        # Intermediate steps.
        L1 = Fs(St, At, Sigma)*dt
        L2 = Fs((St + 0.5*L1), Ak, Sigma)*dt
        L3 = Fs((St + 0.5*L2), Ak, Sigma)*dt
        L4 = Fs((St + L3), Ap, Sigma)*dt

        # NEW "variance" point
        S[t+1] = St + (L1 + 2.0*(L2 + L3) + L4)/6.0
    #_end_for_
    return (m, S)

# End-Of-File
