# Imports:
from numpy import zeros, ones

# Listing 01:
def momGauss(m, S, order=0, dmS=None):
    """
        GAUSSIAN MOMENTS (AND DERIVATIVES)

    [Description]
    This function returns the uncentered moments of a univariate Gaussian
    density, up to 8-th order. Depending on the last argument 'dmS' the
    function also returns the derivative with respect to the marginal means
    'Dm' or the marginal variances 'DS'. An error is thrown if the order of
    the Gaussian moment is not in the [0-8].

    [Input]
    m     : marginal means (N x 1).
    S     : marginal variances (N x 1).
    order : {0,1,2,3,4,5,6,7,8}
    dmS   : {None (default), 'Dm', 'DS'}

    [Output]
    xOut  : uncentered moment (or derivative) of specific 'order' (1 x N).

    Copyright (c) Michail D. Vrettas, PhD - November 2015.

    Last Updated: November 2015.
    """

    # Get the length of 'm'
    N = m.shape[0]

    # Initialize the return variable.
    xOut = None

    # Compute the moment (or derivative) according to the right order.
    if (order == 0):
        xOut = ones((N,1), dtype='float64')
    elif (order == 1):
        if dmS is None:
            xOut = m
        elif dmS == 'Dm':
            xOut = ones((N,1), dtype='float64')
        elif dmS == 'DS':
            xOut = zeros((N,1), dtype='float64')
        # ...
    elif (order == 2):
        if dmS is None:
            xOut = m**2 + S
        elif dmS == 'Dm':
            xOut = 2*m
        elif dmS == 'DS':
            xOut = ones((N,1), dtype='float64')
        # ...
    elif (order == 3):
        if dmS is None:
            xOut = m**3 + 3*m*S
        elif dmS == 'Dm':
            xOut = 3*(m**2 + S)
        elif dmS == 'DS':
            xOut = 3*m
        # ...
    elif (order == 4):
        if dmS is None:
            xOut = m**4 + 6*(m**2)*S + 3*(S**2)
        elif dmS == 'Dm':
            xOut = 4*(m**3 + 3*m*S)
        elif dmS == 'DS':
            xOut = 6*(m**2 + S)
        # ...
    elif (order == 5):
        if dmS is None:
            xOut = m**5 + 10*(m**3)*S + 15*m*(S**2)
        elif dmS == 'Dm':
            xOut = 5*(m**4 + 6*(m**2)*S + 3*(S**2))
        elif dmS == 'DS':
            xOut = 10*(m**3) + 30*(m*S)
        # ...
    elif (order == 6):
        if dmS is None:
            xOut = m**6 + 15*(m**4)*S + 45*(m**2)*(S**2) + 15*(S**3)
        elif dmS == 'Dm':
            xOut = 6*(m**5 + 10*(m**3)*S + 15*m*(S**2))
        elif dmS == 'DS':
            xOut = 15*(m**4) + 90*(m**2)*S + 45*(S**2)
        # ...
    elif (order == 7):
        if dmS is None:
            xOut = m**7 + 21*(m**5)*S + 105*(m**3)*(S**2) + 105*m*(S**3)
        elif dmS == 'Dm':
            xOut = 7*(m**6 + 15*(m**4)*S + 45*(m**2)*(S**2) + 15*(S**3))
        elif dmS == 'DS':
            xOut = 21*(m**5) + 210*(m**3)*S + 315*m*(S**2)
        # ...
    elif (order == 8):
        if dmS is None:
            xOut = m**8 + 28*(m**6)*S + 210*(m**4)*(S**2) + 420*(m**2)*(S**3) + 105*(S**4)
        elif dmS == 'Dm':
            xOut = 8*(m**7 + 21*(m**5)*S + 105*(m**3)*(S**2) + 105*m*(S**3))
        elif dmS == 'DS':
            xOut = 28*(m**6) + 420*(m**4)*S + 1260*(m**2)*(S**2) + 420*(S**3)
        # ...
    else:
        raise ValueError('Unknown Higher Order Gaussian Moment!')

    # --->
    return xOut

# End-Of-File
