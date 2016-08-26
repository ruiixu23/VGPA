""" 
    Backward ODE integrators.
	
    This module implements a set of backward integration methods for the
    computation of the Lagrange multipliers lam(t) and Psi(t) of the variational
    algorithms. To make it easier and more broadly applicable the algorithm
    detects the dimensions of the state vector (D) and calls either the 1D or nD
    version of the selected algorithm. This is because there are signficant
    performance gains when writing the code in 1D rather than nD. Default values
    exist in the integration method ('Euler') and time step ('dt = 0.01') to
    prevent common errors.
	
	List:
            The module include the following methods in both 1D and nD versions
            (a detailed description is given at the individuals methods):
            
            1) Euler (1st Order)
            2) Heun  (2nd Order)
            3) Runge-Kutta (2nd Order)
            4) Runge-Kutta (4th Order)
	
	Author:
		Michail D. Vrettas, PhD
	
	Contact:
		If you find any bug, please contact me at: <vrettasm@gmail.com>
    
	Webpage:
		The code can be downloaded from:
                    <http://vrettasm.weebly.com/software.html>
	
	Copyright (c) Michail D. Vrettas, PhD - November 2015.
"""

# Import ONLY the functions you need.
from numpy import zeros

# Public functions:
__all__ = ['solver']

# Define (lambda) functions for the 'nD' Lagrange multipliers.
fun_lam = lambda dEdm, At, lamt: (-dEdm + lamt.dot(At.T))
fun_Psi = lambda dEdS, At, Psit: (-dEdS + Psit.dot(At) + At.T.dot(Psit))

# Define (lambda) functions for the '1D' Lagrange multipliers.
F_lam = lambda dEdm, At, lamt: (-dEdm + At*lamt)
F_Psi = lambda dEdS, At, Psit: (-dEdS + 2.0*Psit*At)
                         
# Main (callable) method.
def solver(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt=0.01, method='EULER'):
    # Check whether the system is 1D or nD
    if (len(A.shape) == 1):
        nD = 1
    else:
        nD = A.shape[1]
    
    # Switch to the requested algorithm.
    if (method.upper()=='EULER'):
        # In the 1-D case call a faster version.
        if (nD == 1):
            return euler_1D(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt)
        else:
            return euler_nD(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt)
    elif (method.upper()=='HEUN'):
        # In the 1-D case call a faster version.
        if (nD == 1):
            return heun_1D(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt)
        else:
            return heun_nD(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt)
    elif (method.upper()=='RK2'):
        # In the 1-D case call a faster version.
        if (nD == 1):
            return rk2_1D(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt)
        else:
            return rk2_nD(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt)
    elif (method.upper()=='RK4'):
        # In the 1-D case call a faster version.
        if (nD == 1):
            return rk4_1D(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt)
        else:
            return rk4_nD(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt)
    else:
        raise ValueError('Unknown method of integration')

# Function: Euler
def euler_nD(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt):
    """ Euler integration method.
    
        This function implements a simple Euler integration method.
        
        Args:
          A        (array N x D x D): Linear variational parameters
          dEsde_dm (array N x D x 1): derivative of Esde w.r.t m(t)
          dEsde_dS (array N x D x D): derivative of Esde w.r.t S(t)
          dEobs_dm (array N x d x d): derivative of Eobs w.r.t m(t)
          dEobs_dS (array N x d x d): derivative of Eobs w.r.t S(t)
          dt       (float): time (integration) step
        
        Returns:
          lam: Lagrange multipliers for the mean  values (N x D)
          Psi: Lagrange multipliers for the covar values (N x D x D)
          
          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """
    
    # Get the dimensions.
    # D: is the number of the system states
    # N: is the number of discrete time points
    N, D = dEsde_dm.shape
    
    # Preallocate the return vector/matrix.
    lam = zeros((N,D),  dtype='float64')
    Psi = zeros((N,D,D),dtype='float64')

    # Run through all time points.
    for t in range(N-1,0,-1):
        # Get the values at time 't'.
        At = A[t,:,:]
        lamt = lam[t,:]
        Psit = Psi[t,:,:]
        
        # -Eq(14)- NEW "Lamda" point.
        lam[t-1,:] = lamt - fun_lam(dEsde_dm[t,:],At,lamt)*dt + dEobs_dm[t-1,:]
        
        # -Eq(15)- NEW "Psi" point.
        Psi[t-1,:,:] = Psit - fun_Psi(dEsde_dS[t,:,:],At,Psit)*dt + dEobs_dS[t-1,:,:]
    #_end_for_
    return (lam, Psi)

# Function: Euler
def euler_1D(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt):
    """ Euler integration method.
    
        Same as euler_nD but for 1D systems.
        
        Args:
          A        (array N x 1): Linear variational parameters
          dEsde_dm (array N x 1): derivative of Esde w.r.t m(t)
          dEsde_dS (array N x 1): derivative of Esde w.r.t S(t)
          dEobs_dm (array N x 1): derivative of Eobs w.r.t m(t)
          dEobs_dS (array N x 1): derivative of Eobs w.r.t S(t)
          dt       (float): time (integration) step
        
        Returns:
          lam: Lagrange multipliers for the mean  values (N x 1)
          Psi: Lagrange multipliers for the covar values (N x 1)
          
          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """
    
    # Get 'N': the number of discrete time points
    N = dEsde_dm.shape[0]
    
    # Preallocate the return vector/matrix.
    lam = zeros((N,1), dtype='float64')
    Psi = zeros((N,1), dtype='float64')
    
    # Run through all time points.
    for t in range(N-1,0,-1):
        # Get the values at time 't'.
        At = A[t]
        lamt = lam[t]
        Psit = Psi[t]
        
        # -Eq(14)- NEW "Lamda" point.
        lam[t-1] = lamt - F_lam(dEsde_dm[t],At,lamt)*dt + dEobs_dm[t-1]
        
        # -Eq(15)- NEW "Psi" point.
        Psi[t-1] = Psit - F_Psi(dEsde_dS[t],At,Psit)*dt + dEobs_dS[t-1]
    #_end_for_
    return (lam, Psi)


# Function: Heun
def heun_nD(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt):
    """ Heun integration method.
    
        This function implements a simple Heun integration method.
        
        Args:
          A        (array N x D x D): Linear variational parameters
          dEsde_dm (array N x D x 1): derivative of Esde w.r.t m(t)
          dEsde_dS (array N x D x D): derivative of Esde w.r.t S(t)
          dEobs_dm (array N x d x d): derivative of Eobs w.r.t m(t)
          dEobs_dS (array N x d x d): derivative of Eobs w.r.t S(t)
          dt       (float): time (integration) step
        
        Returns:
          lam: Lagrange multipliers for the mean  values (N x D)
          Psi: Lagrange multipliers for the covar values (N x D x D)
          
          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """
    
    # Get the dimensions.
    # D: is the number of the system states
    # N: is the number of discrete time points
    N, D = dEsde_dm.shape
    
    # Preallocate the return vector/matrix.
    lam = zeros((N,D),  dtype='float64')
    Psi = zeros((N,D,D),dtype='float64')
    
    # Half stepsize.
    h = 0.5*dt
    
    # Run through all time points.
    for t in range(N-1,0,-1):
        # Get the values at time 't'.
        At = A[t,:,:]
        lamt = lam[t,:]
        Psit = Psi[t,:,:]
        
        # Get the value at time 't-1'.
        Ak = A[t-1,:,:]
        
        # -Eq(14)- "Lamda" Prediction step.
        ftp = fun_lam(dEsde_dm[t,:],At,lamt)
        
        # "Lamda" Correction step.
        ftc = fun_lam(dEsde_dm[t-1,:],Ak,(lamt - ftp*dt))
        
        # NEW "Lamda" point.
        lam[t-1,:] = lamt - h*(ftp + ftc) + dEobs_dm[t-1,:]
        
        # -Eq(15)- "Psi" Prediction step.
        ftp = fun_Psi(dEsde_dS[t,:,:],At,Psit)
        
        # "Psi" Correction step.
        ftc = fun_Psi(dEsde_dS[t-1,:,:],Ak,(Psit - ftp*dt))
        
        # NEW "Psi" point:
        Psi[t-1,:,:] = Psit - h*(ftp + ftc) + dEobs_dS[t-1,:,:]
    #_end_for_
    return (lam, Psi)

# Function: Heun
def heun_1D(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt):
    """ Heun integration method.
    
        Same as heun_nD but for 1D systems.
        
        Args:
          A        (array N x 1): Linear variational parameters
          dEsde_dm (array N x 1): derivative of Esde w.r.t m(t)
          dEsde_dS (array N x 1): derivative of Esde w.r.t S(t)
          dEobs_dm (array N x 1): derivative of Eobs w.r.t m(t)
          dEobs_dS (array N x 1): derivative of Eobs w.r.t S(t)
          dt       (float): time (integration) step
        
        Returns:
          lam: Lagrange multipliers for the mean  values (N x 1)
          Psi: Lagrange multipliers for the covar values (N x 1)
          
          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """
    
    # Get the dimensions.
    # N: is the number of discrete time points
    N = dEsde_dm.shape[0]
    
    # Preallocate the return vector/matrix.
    lam = zeros((N,1), dtype='float64')
    Psi = zeros((N,1), dtype='float64')
    
    # Half stepsize.
    h = 0.5*dt
    
    # Run through all time points.
    for t in range(N-1,0,-1):
        # Get the values at time 't'.
        At = A[t]
        lamt = lam[t]
        Psit = Psi[t]
        
        # Get the value at time 't-1'.
        Ak = A[t-1]
        
        # -Eq(14)- "Lamda" Prediction step.
        ftp = F_lam(dEsde_dm[t], At, lamt)
        
        # "Lamda" Correction step.
        ftc = F_lam(dEsde_dm[t-1], Ak, (lamt - ftp*dt))
        
        # NEW "Lamda" point.
        lam[t-1] = lamt - h*(ftp + ftc) + dEobs_dm[t-1]
        
        # -Eq(15)- "Psi" Prediction step.
        ftp = F_Psi(dEsde_dS[t], At, Psit)
        
        # "Psi" Correction step.
        ftc = F_Psi(dEsde_dS[t-1], Ak, (Psit - ftp*dt))
        
        # NEW "Psi" point:
        Psi[t-1] = Psit - h*(ftp + ftc) + dEobs_dS[t-1]
    #_end_for_
    return (lam, Psi)


# Function: Runge-Kutta 2
def rk2_nD(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt):
    """ Runge-Kutta integration method.
    
        This function implements a Runge-Kutta 2nd order integration method.
        
        Args:
          A        (array N x D x D): Linear variational parameters
          dEsde_dm (array N x D x 1): derivative of Esde w.r.t m(t)
          dEsde_dS (array N x D x D): derivative of Esde w.r.t S(t)
          dEobs_dm (array N x d x d): derivative of Eobs w.r.t m(t)
          dEobs_dS (array N x d x d): derivative of Eobs w.r.t S(t)
          dt       (float): time (integration) step
        
        Returns:
          lam: Lagrange multipliers for the mean  values (N x D)
          Psi: Lagrange multipliers for the covar values (N x D x D)
          
          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """
    # Get the dimensions.
    # D: is the number of the system states
    # N: is the number of discrete time points
    N, D = dEsde_dm.shape
    
    # Preallocate the return vector/matrix.
    lam = zeros((N,D) , dtype='float64')
    Psi = zeros((N,D,D),dtype='float64')
    
    # Half stepsize.
    h = 0.5*dt
    
    # Run through all time points.
    for t in range(N-1,0,-1):
        # Get the values at time 't'.
        At = A[t,:,:]
        lamt = lam[t,:]
        Psit = Psi[t,:,:]
        dEsde_dmt = dEsde_dm[t,:]
        dEsde_dSt = dEsde_dS[t,:,:]
        
        # Get the midpoints at time 't - 0.5*dt'.
        Ak = 0.5*(A[t-1,:,:] + At)
        dEmk = 0.5*(dEsde_dm[t-1,:] + dEsde_dmt)
        dESk = 0.5*(dEsde_dS[t-1,:,:] + dEsde_dSt)
        
        # Lamda (backward) propagation.
        lamk = lamt - h*fun_lam(dEsde_dmt, At, lamt)
        
        # NEW "lamda" point.
        lam[t-1,:] = lamt - fun_lam(dEmk, Ak, lamk)*dt + dEobs_dm[t-1,:]
        
        # Psi (backward) propagation.
        Psik = Psit - h*fun_Psi(dEsde_dSt, At, Psit)
    
        # NEW "Psi" point.
        Psi[t-1,:,:] = Psit - fun_Psi(dESk, Ak, Psik)*dt + dEobs_dS[t-1,:,:]
    #_end_for_
    return (lam, Psi)


# Function: Runge-Kutta 2
def rk2_1D(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt):
    """ Runge-Kutta integration method.
    
        Same as rk2_nD but for 1D systems.
        
        Args:
          A        (array N x 1): Linear variational parameters
          dEsde_dm (array N x 1): derivative of Esde w.r.t m(t)
          dEsde_dS (array N x 1): derivative of Esde w.r.t S(t)
          dEobs_dm (array N x 1): derivative of Eobs w.r.t m(t)
          dEobs_dS (array N x 1): derivative of Eobs w.r.t S(t)
          dt       (float): time (integration) step
        
        Returns:
          lam: Lagrange multipliers for the mean  values (N x D)
          Psi: Lagrange multipliers for the covar values (N x D x D)
          
          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """
    # Get the dimensions.
    # N: is the number of discrete time points
    N = dEsde_dm.shape[0]
    
    # Preallocate the return vector/matrix.
    lam = zeros((N,1), dtype='float64')
    Psi = zeros((N,1), dtype='float64')
    
    # Half stepsize.
    h = 0.5*dt
    
    # Run through all time points.
    for t in range(N-1,0,-1):
        # Get the values at time 't'.
        At = A[t]
        lamt = lam[t]
        Psit = Psi[t]
        dEsde_dmt = dEsde_dm[t]
        dEsde_dSt = dEsde_dS[t]
        
        # Get the midpoints at time 't - 0.5*dt'.
        Ak = 0.5*(A[t-1] + At)
        dEmk = 0.5*(dEsde_dm[t-1] + dEsde_dmt)
        dESk = 0.5*(dEsde_dS[t-1] + dEsde_dSt)
        
        # Lamda (backward) propagation.
        lamk = lamt - h*F_lam(dEsde_dmt, At, lamt)
        
        # NEW "lamda" point.
        lam[t-1] = lamt - F_lam(dEmk, Ak, lamk)*dt + dEobs_dm[t-1]
        
        # Psi (backward) propagation.
        Psik = Psit - h*F_Psi(dEsde_dSt, At, Psit)
    
        # NEW "Psi" point.
        Psi[t-1] = Psit - F_Psi(dESk, Ak, Psik)*dt + dEobs_dS[t-1]
    #_end_for_
    return (lam, Psi)

# Function: Runge-Kutta 4
def rk4_nD(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt):
    """ Runge-Kutta integration method.
    
        This function implements a Runge-Kutta 4th order integration method.
        
        Args:
          A        (array N x D x D): Linear variational parameters
          dEsde_dm (array N x D x 1): derivative of Esde w.r.t m(t)
          dEsde_dS (array N x D x D): derivative of Esde w.r.t S(t)
          dEobs_dm (array N x d x d): derivative of Eobs w.r.t m(t)
          dEobs_dS (array N x d x d): derivative of Eobs w.r.t S(t)
          dt       (float): time (integration) step
        
        Returns:
          lam: Lagrange multipliers for the mean  values (N x D)
          Psi: Lagrange multipliers for the covar values (N x D x D)
          
          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """
    
    # Get the dimensions.
    # D: is the number of the system states
    # N: is the number of discrete time points
    N, D = dEsde_dm.shape
    
    # Preallocate the return vector/matrix.
    lam = zeros((N,D),  dtype='float64')
    Psi = zeros((N,D,D),dtype='float64')
    
    # Runge-Kutta 4 (4th-Order).
    for t in range(N-1,0,-1):
        # Get the values at time 't'.
        At = A[t,:,:]
        lamt = lam[t,:]
        Psit = Psi[t,:,:]
        dEsde_dmt = dEsde_dm[t,:]
        dEsde_dSt = dEsde_dS[t,:,:]
        
        # Get the (previous) value at time 't-1'.
        Ap = A[t-1,:,:]
        dEsde_dmp = dEsde_dm[t-1,:]
        dEsde_dSp = dEsde_dS[t-1,:,:]
        
        # Get the midpoints at time 't - 0.5*dt'.
        Ak = 0.5*(Ap + At)
        dEmk = 0.5*(dEsde_dmp + dEsde_dmt)
        dESk = 0.5*(dEsde_dSp + dEsde_dSt)
        
        # Lamda (backward) propagation: Intermediate steps.
        K1 = fun_lam(dEsde_dmt, At, lamt)*dt
        K2 = fun_lam(dEmk, Ak, (lamt - 0.5*K1))*dt
        K3 = fun_lam(dEmk, Ak, (lamt - 0.5*K2))*dt
        K4 = fun_lam(dEsde_dmp, Ap,(lamt - K3))*dt
        
        # NEW "lamda" point.
        lam[t-1,:] = lamt - (K1 + 2.0*(K2 + K3) + K4)/6.0 + dEobs_dm[t-1,:]
        
        # Psi (backward) propagation: Intermediate steps.
        L1 = fun_Psi(dEsde_dSt, At, Psit)*dt
        L2 = fun_Psi(dESk, Ak, (Psit - 0.5*L1))*dt
        L3 = fun_Psi(dESk, Ak, (Psit - 0.5*L2))*dt
        L4 = fun_Psi(dEsde_dSp, Ap,(Psit - L3))*dt
        
        # NEW "Psi" point.
        Psi[t-1,:,:] = Psit - (L1 + 2.0*(L2 + L3) + L4)/6.0 + dEobs_dS[t-1,:,:]
    #_end_for_
    return (lam, Psi)

# Function: Runge-Kutta 4
def rk4_1D(A, dEsde_dm, dEsde_dS, dEobs_dm, dEobs_dS, dt):
    """ Runge-Kutta integration method.
    
        Same as rk4_nD but for 1D systems.
        
        Args:
          A        (array N x 1): Linear variational parameters
          dEsde_dm (array N x 1): derivative of Esde w.r.t m(t)
          dEsde_dS (array N x 1): derivative of Esde w.r.t S(t)
          dEobs_dm (array N x 1): derivative of Eobs w.r.t m(t)
          dEobs_dS (array N x 1): derivative of Eobs w.r.t S(t)
          dt       (float): time (integration) step
        
        Returns:
          lam: Lagrange multipliers for the mean  values (N x 1)
          Psi: Lagrange multipliers for the covar values (N x 1)
          
          The return vector/matrix are 'None' if an error has occurred
          while integrating the ODEs.
    """
    
    # Get the dimensions.
    # N: is the number of discrete time points
    N = dEsde_dm.shape[0]
    
    # Preallocate the return vector/matrix.
    lam = zeros((N,1), dtype='float64')
    Psi = zeros((N,1), dtype='float64')
    
    # Runge-Kutta 4 (4th-Order).
    for t in range(N-1,0,-1):
        # Get the values at time 't'.
        At = A[t]
        lamt = lam[t]
        Psit = Psi[t]
        dEsde_dmt = dEsde_dm[t]
        dEsde_dSt = dEsde_dS[t]
        
        # Get the (previous) value at time 't-1'.
        Ap = A[t-1]
        dEsde_dmp = dEsde_dm[t-1]
        dEsde_dSp = dEsde_dS[t-1]
        
        # Get the midpoints at time 't - 0.5*dt'.
        Ak = 0.5*(Ap + At)
        dEmk = 0.5*(dEsde_dmp + dEsde_dmt)
        dESk = 0.5*(dEsde_dSp + dEsde_dSt)
        
        # Lamda (backward) propagation: Intermediate steps.
        K1 = F_lam(dEsde_dmt, At, lamt)*dt
        K2 = F_lam(dEmk, Ak, (lamt - 0.5*K1))*dt
        K3 = F_lam(dEmk, Ak, (lamt - 0.5*K2))*dt
        K4 = F_lam(dEsde_dmp, Ap,(lamt - K3))*dt
        
        # NEW "lamda" point.
        lam[t-1] = lamt - (K1 + 2.0*(K2 + K3) + K4)/6.0 + dEobs_dm[t-1]
        
        # Psi (backward) propagation: Intermediate steps.
        L1 = F_Psi(dEsde_dSt, At, Psit)*dt
        L2 = F_Psi(dESk, Ak, (Psit - 0.5*L1))*dt
        L3 = F_Psi(dESk, Ak, (Psit - 0.5*L2))*dt
        L4 = F_Psi(dEsde_dSp, Ap,(Psit - L3))*dt
        
        # NEW "Psi" point.
        Psi[t-1] = Psit - (L1 + 2.0*(L2 + L3) + L4)/6.0 + dEobs_dS[t-1]
    #_end_for_
    return (lam, Psi)

# End-Of-File
