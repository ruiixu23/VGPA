# Numerical:
import numpy as np

# Variational:
from core.varFreeEnergy import varFreeEnergy
from core.gradL_Ab import gradL_Ab

# "Numerical/auxiliary" (local):
from auxiliary.numerical import finiteDiff
from auxiliary.optimize import optim_SCG

# Public functions:
__all__ = ['smoothing']

# Listing: 01
def smoothing(fun_sde, x0, m0, S0, sde_struct, nit = 150):
    """
        SMOOTHING
    
    [Description]:
    Variational Gaussian Process Approximation (VGPA) for inference of SDEs.
    The smoothing process is performed within a predefined time interval [t0-tf]
    given a discrete set of noisy observations. The optimisation is performed
    with an adaptive scaled-conjugate-gradient optimization algorithm (SCG).
    
    [Input Parameters]:
    fun_sde : Objective function handle. This is model specific and has to be
              given for each dynamical system separately.
    x0      : Initial search point.
    m0      : Initial (posterior) mean state (1 x D).
    S0      : Initial (posterior) covariance (D x D).
    sde_struct : Data structure (dictionary) that holds model parameters.
    nit     : Maximum number of iterations.
    
    [Output Parameters]:
    F      : Variational Free energy (scalar value).
    mParam : Output parameters include:
            (1) m   : Posterior means (N x D).
            (2) S   : Posterior covariances (N x D x D).
            (3) A   : Variational linear parameters (N x D x D).
            (4) b   : Variational offset parameters (N x D).
            (5) lam : Lagrange multipliers for m-constraint (N x D).
            (6) Psi : Lagrange multipliers for S-constraint (N x D x D).
            (7) stat: Statistics from optimization.
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Updated: November 2015.
    """
    
    # Get the dimensions of the problem.
    D = sde_struct['D']
    N = sde_struct['N']
    
    # Setup the model parameters. This data structure will pass in
    # both: (1) fun_sde, and (2) fun_grad functions and then it is
    # our responsibility to extract and use the correct parameters
    # inside each function.
    mParam = {'m0':m0, 'S0':S0, 'fun':fun_sde, 'sde_struct':sde_struct}
    
    # Check numerically the gradients.
    if sde_struct['checkGradf']:
        print("BEFORE-OPT")
        grad_A = gradL_Ab(x0, mParam, True)
        grad_N = finiteDiff(varFreeEnergy, x0, mParam)
        print(" MAE: {0}".format(np.abs(grad_A-grad_N).sum()/grad_N.size))
    # ...
    
    # Setup SCG options.
    options = {'nit':nit, 'xtol':1.0e-6, 'ftol':1.0e-8,\
               'disp':True, 'lmin':True}
    
    # My SCG optimization routine.
    x, F, mParam, stat = optim_SCG(varFreeEnergy, x0, gradL_Ab, options, mParam)
    
    # Check numerically the gradients.
    if sde_struct['checkGradf']:
        print("AFTER-OPT")
        grad_A = gradL_Ab(x, mParam, True)
        grad_N = finiteDiff(varFreeEnergy, x, mParam)
        print(" MAE: {0}".format(np.abs(grad_A-grad_N).sum()/grad_N.size))
    # ...
    
    # Unpack data.
    if (D == 1):
        A = x[:N]
        b = x[N:]
    else:
        # Total number of linear parameters 'A'.
        K = D*D*N
        # Reshape them before return.
        A = x[:K].reshape(N,D,D)
        b = x[K:].reshape(N,D)
    # ...
    
    # Update the structure with the (final) var. parameters.
    mParam['At'] = A
    mParam['bt'] = b
    
    # Update the structure with the optimization statistics.
    mParam['stat'] = stat
    
    # --->
    return F, mParam

# End-Of-File