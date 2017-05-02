# Imports:
import numpy as np

# Public functions:
__all__ = ['collect_obs']

# Listing 01:
def collect_obs(xt, T, n_Obs=1, R=0.2, hMask=None):
    """
        COLLECT_OBS

    [Description]
    This function collects a number of noisy observations from a sample path
    (trajectory). The observations are collected at equidistant time points.

    [Input]
    xt    : Discretised trajectory (sample path) [N x D].
    T     : Time window of inference [N x 1].
    n_Obs : Observations density (# of observations per time unit).
    R     : Observation noise (co)variance.
    hMask : List that masks only the observed values.

    [Output]
    obsX : observation times
    obsY : observation values (with i.i.d. white noise)

    SEE ALSO : create_system_paths.

    Copyright (c) Michail D. Vrettas, PhD - November 2015.

    Last Updated: June 2016.
    """

    # Get the current random seed.
    r0 = np.random.get_state()

    # Set the current random seed.
    np.random.seed(2015)

    # Get the discretization step.
    dt = np.diff(T)[0]

    # Check if the required number of observations
    # per time unit exceeds the available capacity
    # of samples.
    if (n_Obs > 1.0/dt):
        raise ValueError('Observation density exceeds the number of samples.')

    # Total number of observations.
    M = int(np.floor(np.abs(T[0]-T[-1])*n_Obs))

    # Number of discretized time points.
    N = T.shape[0]

    # Observation indeces.
    idx = np.linspace(0, N, M+2, dtype=np.int)

    # Convert it to list so you can use it as index.
    obsX = np.array(idx[1:-1].tolist())

    # Extract the complete observations (d = D) at times obsX.
    obsY = np.take(xt, obsX, axis=0)

    # Check if a mask has been given.
    if hMask is not None:
        # Here we have (d < D)
        obsY = obsY[:,hMask]

    # Dimensionality of observations.
    d = obsY.shape[1]

    # Check if (co)variance vector/matrix is given.
    if np.isscalar(R):
        # Add fixed Gaussian noise.
        obsY = obsY + np.sqrt(R)*np.random.randn(M,1)
    else:
        # Variance (diagonal) matrix.
        if (len(R.shape) == 1):
            R = np.diag(R)

        # Get the square root of the noise matrix.
        # Here sqrt() works because we assume that
        # R is diagonal.
        sR = np.sqrt(R)

        # Add fixed Gaussian noise.
        obsY = obsY + sR.dot(np.random.randn(d,M)).T

    # Restore the random seed value.
    np.random.set_state(r0)

    # --->
    return obsX, obsY

# End-Of-File
