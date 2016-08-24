# Imports
import numpy as np
from copy import deepcopy

# Public functions:
__all__ = ['optim_SCG', 'optim_CG', 'optim_SA']

# Listing: 01
def optim_SCG(f, x0, gf, options, mParam):
    """
        OPTIM_SCG:
    
    [Description]
    Scaled conjugate gradient optimization, attempts to find a local minimum
    of the function f(x). Here 'x0' is a column vector and 'f' returns a scalar
    value. The minimisation process uses also the gradient 'gf' (i.e. df(x)/dx).
    The point at which 'f' has a local minimum is returned as 'x'. The function
    value at that point (the minimum) is returned in "fx".
    
    [Input parameres]:
    f      : is the objective function to be optimised.
    x0     : is the initial point of function (D x 1).
    gf     : is the derivative of the objective function w.r.t 'x'.
    mParam : is a dictionary containing all additional parameters for both
             'f' and 'gf' functions.
    options: is an set of optional values for the optimizer:
                (1) the number of iterations,
                (2) absolute precision in ' x',
                (3) absolute precision in 'fx',
                (4) display error statistics flag
    
    [Output parameters]:
    x      : the point where the minimum was found.
    fx     : the function value, at the minimum point.
    mParam : additional output parameters for "f".
    stat   : statistics that collected through the optimisation process.
    
    NOTE: This code is adopted from NETLAB (a free MATLAB library)
    
    Reference Book:
    (1) Ian T. Nabney (2001): Netlab: Algorithms for Pattern Recognition.
    Advances in Pattern Recognition, Springer.
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Updated: November 2015.
    """
    
    # Maximum number of iterations.
    if 'nit' in options:
        nit = options['nit']
    else:
        nit = 150
    
    # Error tolerance in 'x'.
    if 'xtol' in options:
        xtol = options['xtol']
    else:
        xtol = 1.0e-6
    
    # Error tolerance in 'fx'.
    if 'ftol' in options:
        ftol = options['ftol']
    else:
        ftol = 1.0e-8
    
    # Display statistics flag.
    if 'disp' in options:
        disp = options['disp']
    else:
        disp = False
    
    # Display method name.
    print(" >> SCG optimization")
    
    # Dimensionality of the input vector.
    D = x0.size
    
    # Preallocate stats dictionary.
    stat = {'MaxIt':nit, 'Fx':np.zeros((nit,1)), 'f_eval':0, 'g_eval':0, \
            'beta':np.zeros((nit,1)), 'gradx':np.zeros((nit,1))}
    
    # Initialization
    x = deepcopy(x0)
    
    # Make sure input is a column-vector.
    x = x.reshape(D,1)
    
    # Initial sigma value.
    sigma0 = 1.0e-3
    
    # Initial function/gradients value.
    fnow = f(x, mParam)
    gradnew = gf(x, mParam)
    
    # Increase function/gradient evaluations by one.
    stat['f_eval'] += 1
    stat['g_eval'] += 1
    
    # Store the current values.
    fold = fnow
    
    # Store the current gradient.
    gradold = gradnew
    
    # Setup the initial search direction.
    d = -gradnew
    
    # Force calculation of directional derivatives.
    success = 1
    
    # Counts the number of successes.
    nsuccess = 0
    
    # Initial scale parameter.
    beta = 1.0
    
    # Lower & Upper bounds on scale (beta).
    betaMin = 1.0e-15; betaMax = 1.0e+100
    
    # Get the machine precision constant.
    eps_float = np.finfo(float).eps
    
    # Main optimization loop.
    for j in range(nit):
        # Calculate first and second directional derivatives.
        if (success == 1):
            mu = d.T.dot(gradnew)
            if (mu >= 0.0):
                d = -gradnew
                mu = d.T.dot(gradnew)
            
            # Compute kappa and check for termination.
            kappa = d.T.dot(d)
            if (kappa < eps_float):
                fx = fnow; stat['MaxIt'] = j
                return x, fx, mParam, stat
            
            # Update sigma and check the gradient on a new direction.
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma*d
            
            # Because we evaluate only gf(xplus) we set the flag 'True'.
            gplus = gf(xplus, mParam, True)
            
            # Increase function/gradients evaluations by one.
            stat['f_eval'] += 1
            stat['g_eval'] += 1
            
            # Compute theta.
            theta = (d.T.dot(gplus - gradnew))/sigma
        # ...
        
        # Increase effective curvature and evaluate step size alpha.
        delta = theta + beta*kappa
        if (delta <= 0):
            delta = beta*kappa
            beta = beta-(theta/kappa)
        # ...
        alpha = -(mu/delta)
        
        # Evaluate the function at a new point.
        xnew = x + alpha*d
        fnew = f(xnew, mParam)
        stat['f_eval'] += 1
        
        # Calculate the new comparison ratio.
        Delta = 2.0*(fnew - fold)/(alpha*mu)
        if (Delta >= 0):
            success = 1
            nsuccess += 1
            x = xnew
            fnow = fnew
            gnow = gradnew
        else:
            success = 0
            fnow = fold
            gnow = gradold
        # ...
        
        # Total gradient.
        totGrad = np.math.fsum(np.abs(gnow))
        
        # Store statistics
        stat['Fx'][j] = fnow
        stat['beta'][j] = beta
        stat['gradx'][j] = totGrad
        
        # Used in debuging mode.
        if disp:
            print(' {0}:\tfx={1:.3f}\tsum(gx)={2:.3f}'.format(j,fnow,totGrad))
        
        # ...
        if (success == 1):
            # Check for termination.
            if (np.abs(alpha*d).max() <= xtol) and (np.abs(fnew-fold) <= ftol):
                fx = fnew; stat['MaxIt'] = j
                return x, fx, mParam, stat
            else:
                # Update variables for new position.
                fold = fnew
                gradold = gradnew
                
                # Evaluate function/gradient at the new point.
                fnow = f(x, mParam)
                gradnew = gf(x, mParam)
                
                #  Increase function/gradients evaluations by one.
                stat['f_eval'] += 1
                stat['g_eval'] += 1
                
                # If the gradient is zero then we are done.
                if (gradnew.T.dot(gradnew) == 0.0):
                    fx = fnow; stat['MaxIt'] = j
                    return x, fx, mParam, stat
            # ...
        # ...
        
        # Adjust beta according to comparison ratio.
        if (Delta < 0.25):
            beta = min(4.0*beta, betaMax)
        
        if (Delta > 0.75):
            beta = max(0.5*beta, betaMin)
        
        # Update search direction using Polak-Ribiere formula,
        # or re-start in direction of negative gradient after
        # 'D' steps.
        if (nsuccess == D):
            d = -gradnew
            nsuccess = 0
        else:
            if (success == 1):
                gamma = max(gradnew.T.dot(gradold - gradnew)/mu, 0.0)
                d = gamma*d - gradnew
    # end-for
    
    # Display a warning to the user.
    print(' Maximum number of iterations has been reached.')
    
    # Here we have reached the maximum number of iterations.
    fx = fold
    
    # --->
    return x, fx, mParam, stat

# Listing: 02
def optim_CG(f, x0, gf, options, mParam):
    """
        OPTIM_CG:
    
    [Description]
    Conjugate gradient optimization, attempts to find a local minimum
    of the function f(x). Here 'x0' is a column vector and 'f' returns
    a scalar value. The minimisation process uses also the gradient 'gf'
    (i.e. df(x)/dx). The point at which 'f' has a local minimum is returned
    as 'x'. The function value at that point (the minimum) is returned in "fx".
    
    [Input parameres]:
    f      : is the objective function to be optimised.
    x0     : is the initial point of function (D x 1).
    gf     : is the derivative of the objective function w.r.t 'x'.
    mParam : is a dictionary containing all additional parameters for both
             'f' and 'gf' functions.
    options: is an set of optional values for the optimizer:
                (1) the number of iterations,
                (2) absolute precision in ' x',
                (3) absolute precision in 'fx',
                (4) display error statistics flag
    
    [Output parameters]:
    x      : the point where the minimum was found.
    fx     : the function value, at the minimum point.
    mParam : additional output parameters for "f".
    stat   : statistics that collected through the optimisation process.
    
    NOTE: This code is adopted from NETLAB (a free MATLAB library)
    
    Reference Book:
    (1) Ian T. Nabney (2001): Netlab: Algorithms for Pattern Recognition.
    Advances in Pattern Recognition, Springer.
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Updated: November 2015.
    """
    
    # Maximum number of iterations.
    if 'nit' in options:
        nit = options['nit']
    else:
        nit = 500
    
    # Error tolerance in 'x'.
    if 'xtol' in options:
        xtol = options['xtol']
    else:
        xtol = 1.0e-6
    
    # Error tolerance in 'fx'.
    if 'ftol' in options:
        ftol = options['ftol']
    else:
        ftol = 1.0e-8
    
    # Display statistics flag.
    if 'disp' in options:
        disp = options['disp']
    else:
        disp = False
    
    # Display method name.
    print(" >> CG optimization (with backtracking)")
    
    # Dimensionality of the input vector.
    D = x0.size
    
    # Preallocate stats dictionary.
    stat = {'MaxIt':nit, 'Fx':np.zeros((nit,1)), 'f_eval':0, 'g_eval':0,\
            'eta':np.zeros((nit,1)), 'gradx':np.zeros((nit,1))}
    
    # Initialization
    x = deepcopy(x0)
    
    # Make sure input is a column-vector.
    x = x.reshape(D,1)
    
    # Initial learning rate: must be positive.
    eta0 = 0.15
    
    # Decrease rate for the step size.
    r = 0.5
    
    # Evaluate function and gradient.
    fnew = f(x, mParam) 
    gradnew = gf(x, mParam)
    
    # Increase function/gradient evaluations by one.
    stat['f_eval'] += 1
    stat['g_eval'] += 1
    
    # Search direction goes downhill.
    d = -gradnew
    
    # Store statistics.
    stat['Fx'][0] = fnew
    stat['eta'][0] = eta0
    stat['gradx'][0] = np.sum(np.abs(gradnew))
    
    # Main optimization loop.
    for j in range(1,nit):
        # Keep the old values.
        xold = x
        fold = fnew
        gradold = gradnew
        
        # Check if the gradient is zero.
        mu = gradold.T.dot(gradold)
        if (mu == 0.0):
            fx = fold; stat['MaxIt'] = j
            return x, fx, mParam, stat
        
        # This shouldn't occur, but rest of code depends on 'd' being downhill.
        if (gradnew.T.dot(d) > 0.0):
            d = -d
        
        # Update search direction.
        line_sd = d/np.linalg.norm(d)
        
        # Try to find optimum stepsize.
        eta, cnt = backtrack(f, xold, fold, line_sd, eta0, r, deepcopy(mParam))
        
        # Update the function evaluations.
        stat['f_eval'] += cnt
        
        # Exit if you can't find any better eta.
        if (eta == 0.0):
            x = xold; fx = fold; stat['MaxIt'] = j
            return x, fx, mParam, stat
        
        # Set x and fnew to be the actual search point we have found.
        x = xold + eta*line_sd
        fnew = f(x, mParam) 
        gradnew = gf(x, mParam)
        
        # Increase function/gradient evaluations by one.
        stat['f_eval'] += 1
        stat['g_eval'] += 1
        
        # Check for termination.
        if (np.abs(x-xold).max() <= xtol) and (np.abs(fnew-fold) <= ftol):
            fx = fnew; stat['MaxIt'] = j
            return x, fx, mParam, stat
        
        # Use Polak-Ribiere formula to update search direction.
        gamma = max(gradnew.T.dot(gradold - gradnew)/mu, 0.0)
        d = gamma*d - gradnew
        
        # Total gradient.
        totGrad = np.math.fsum(np.abs(gradnew))
        
        # Store statistics.
        stat['Fx'][j] = fnew
        stat['eta'][j] = eta
        stat['gradx'][j] = totGrad
        
        # Used in debuging mode.
        if disp:
            print(' {0}:\tfx={1:.3f}\tsum(gx)={2:.3f}'.format(j, fnew, totGrad))
    # ...
    
    # Display a warning to the user.
    print(' Maximum number of iterations has been reached.')
    
    # Here we have reached the maximum number of iterations.
    fx = fold
    
    # --->
    return x, fx, mParam, stat

# Listing: 03
def backtrack(f, x0, f0, df0, eta0, r, *args):
    """
        BACTRACK
    
    Description:
    Backtracking method to find optimum step size
    for the conjugate gradient algorithm (OPTIM_CG).
    
    [Input parameters]:
    f     : is the objective function to be optimised.
    x0    : is the current search point of function (D x 1).
    fx0   : is the value of the objective function 'f',
            at x0, i.e. f0 = f(x0) (1 x 1).
    df0   : is the value of the gradient function 'df',
            at x0, i.e. df0 = df(x0) (D x 1).
    eta0  : current stepsize 0 < eta < 1.
    r     : decrease ratio for step size 0 < r < 1.
    *args : additional parameters for function 'f'.
    
    [Output parameters]:
    eta   : optimal step size.
    cnt   : number of function evaluations.
    
    See also: optim_SCG
    
    Copyright (c) Michail D. Vrettas, PhD (November 2015)
    
    Last Updated: November 2015.
    """
    
    # Maximum number of trials.
    maxiter = 15
    
    # Optimum step size.
    eta = eta0
    
    # Counter.
    cnt = 0
    
    # Evaluate the function.
    fx = f((x0 + eta0*df0), *args)
    
    # Termination condition for backtracking.
    while ((cnt < maxiter) and (not np.isfinite(fx) or (fx > f0))):
        # Decrease stepsize.
        eta *= r
        # Compute the new position.
        x = x0 + eta*df0
        # Evaluate the function.
        fx = f(x,*args)
        # Increase counter by one.
        cnt += 1
    # ...
    
    # Safeguard: --->
    return max(eta, 0.0), cnt+1

# Listing: 03
def optim_SA(f, x0, gf, options, mParam):
    """
        OPTIM_SA:
    
    [Description]
    Simulated annealing optimization, attempts to find a local minimum
    of the function f(x). Here 'x0' is a column vector and 'f' returns
    a scalar value. The minimisation process uses also the gradient 'gf'
    (i.e. df(x)/dx). The point at which 'f' has a local minimum is returned
    as 'x'. The function value at that point (the minimum) is returned in "fx".
    
    [Input parameres]:
    f      : is the objective function to be optimised.
    x0     : is the initial point of function (D x 1).
    gf     : is the derivative of the objective function w.r.t 'x'.
    mParam : is a dictionary containing all additional parameters for both
             'f' and 'gf' functions.
    options: is an set of optional values for the optimizer:
                (1) the number of iterations,
                (2) absolute precision in ' x',
                (3) absolute precision in 'fx',
                (4) display error statistics flag
    
    [Output parameters]:
    x      : the point where the minimum was found.
    fx     : the function value, at the minimum point.
    mParam : additional output parameters for "f".
    stat   : statistics that collected through the optimisation process.
    
    Reference:
    (1) ...
    
    Copyright (c) Michail D. Vrettas, PhD - November 2015.
    
    Last Updated: December 2015.
    """
    
    # Maximum number of runs.
    if 'nrun' in options:
        nrun = options['nrun']
    else:
        nrun = 1
    
    # Error tolerance in 'x'.
    if 'xtol' in options:
        xtol = options['xtol']
    else:
        xtol = 1.0e-6
    
    # Error tolerance in 'fx'.
    if 'ftol' in options:
        ftol = options['ftol']
    else:
        ftol = 1.0e-8
    
    # Display statistics flag.
    if 'disp' in options:
        disp = options['disp']
    else:
        disp = False
    
    # Display method name.
    print(" >> Simulated Annealing optimization (with SCG local search)")
    
    # Dimensionality of the input vector.
    D = x0.size
    
    # Initialization
    x = deepcopy(x0)
    
    # Make sure input is a column-vector.
    x = x.reshape(D,1)
    
    # Maximum number of iterations.
    if 'nit' in options:
        nit = options['nit']
    else:
        nit = 100*D
    
    # Preallocate stats dictionary.
    stat = {'MaxIt':nit, 'Fx':np.zeros((nit,1)), 'f_eval':0, 'g_eval':0,\
            'Temp':np.zeros((nit,1))}
    
    # Local search.
    if 'lmin' in options:
        lmin = options['lmin']
    else:
        lmin = False
    
    # Evaluate function and gradient.
    fx = f(x, mParam) 
    gradf = gf(x, mParam)
    
    # Increase function/gradient evaluations by one.
    stat['f_eval'] += 1
    stat['g_eval'] += 1
    
    # Controls after how many iterations the Temperature will decrease.
    kappa = 10
    
    # Initial / Final Temperature.
    T0 = 1.0; TF = 1.0e-2; Tk = T0
    
    # Termination threshold.
    max_lim = 1000
    
    # Acceptance counter.
    acc = 0
    
    # Restart "nrun" times [optional].
    for n in range(nrun):
        # Display current run value
        print(" Run: {0} ".format(n))
        
        # Main iteration loop.
        for i in range(nit):
            # Update current temperature.
            if (np.mod(i,kappa) == 0.0):
                Tk = (T0 * (TF/T0)**(i/nit))
            
            # Propose new state using normaly distributed
            # numbers with mean "x" (the previous state).
            xnew = x + Tk*np.random.randn(D,1) - (1.0 - Tk)*gradf
            
            # Evaluate function and gradient.
            fnew = f(xnew, mParam) 
            gradfnew = gf(xnew, mParam)
            
            # Increase function/gradient evaluations by one.
            stat['f_eval'] += 1
            stat['g_eval'] += 1
            
            # Calculate difference between the two energy states.
            dFx = fnew-fx
            
            # Check for acceptance.
            if((dFx < 0) or (np.exp(-dFx/Tk) > np.random.rand())):
                # Increase accepted states by one
                acc += 1
                # Update to the new values.
                x = xnew
                fx = fnew
                gradf = gradfnew
            
            # Store statistics.
            stat['Fx'][i] = fx
            stat['Temp'][i] = Tk
            
            # Used in debuging mode.
            if disp:
                print(' {0}:\tfx={1:.3f}\tTemp={2:.3f}\tAcc.Ratio = {3:.2f}%'.\
                                            format(i, fx, Tk, (100*acc/(i+1))))
            
            # Check for termination.
            # If the minimum has not changed for 'max_lim' iterations
            # then break the loop.
            if ((i >= max_lim) and\
                (np.diff(stat['Fx'][i-max_lim:i], axis=0).sum() == 0.0)):
                print(' Algorithm terminated in {0} iterations'.format(i))
                stat['MaxIt'] = i
                break
        
        # Current optimum values (after optimisation).
        xopt = x; fopt = fx
        
        # From the final result; perform local minimization using SCG.
        if (lmin):
            print(" >> Searching locally using SCG ...")
            # Setup local optimisation parameters.
            local_options = {'nit':1000, 'xtol':xtol, 'ftol':ftol, 'disp':True}
            # Call SCG for a small number of iterations.
            x_loc, fx_loc, mParam_loc, _ =\
                        optim_SCG(f, xopt, gf, local_options, deepcopy(mParam))
            
            # If this local minimum is better; then accept it.
            if(fx_loc < fopt):
                xopt = x_loc
                fopt = fx_loc
                mParam = mParam_loc
        
    # --->
    return xopt, fopt, mParam, stat

# End-of-file.