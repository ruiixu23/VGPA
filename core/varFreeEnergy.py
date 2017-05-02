# Imports:
from numpy import asscalar
from core import fwd_ode, bwd_ode, likelihood, kl0

# Public functions:
__all__ = ['varFreeEnergy']

# Listing: 01
def varFreeEnergy(Ab, mParam):
    """
        VARIATIONAL FREE ENERGY (COST FUNCTION)

    [Description]
    Computes parameters related to the variational posterior process
    defined by the linear/offset parameters A(t) and b(t).

    [Input]
    Ab     : initial variational linear parameters 'A' (N x D x D) +
             initial variational offset parameters 'b' (N x D).
    mParam : structure that holds all the parameters.

    [Output]
    F      : variational Free Energy.

    N.B: Since the structure 'mParam' is passed by reference all the
         parameters it contains are updated and returned implicitly.

    SEE ALSO: kl0, likelihood, fwd_ode, bwd_ede.

    NOTE: The equation numbers correspond to the paper:

    @CONFERENCE{Archambeau2007b,
      author = {Cedric Archambeau, Manfred Opper, Yuan Shen, Dan Cornford and
      J. Shawe-Taylor}, title = {Variational Inference for Diffusion Processes},
      booktitle = {Annual Conference on Neural Information Processing Systems},
      year = {2007}
    }

    Copyright (c) Michail D. Vrettas - November 2015.

    Last Updated: November 2015.
    """

    # Extract SDE parameters (dictionary structure).
    sde_struct = mParam['sde_struct']

    # Integration method for the ODEs.
    ode_method = sde_struct['ode_method']

    # System (diffusion) noise.
    Sigma = sde_struct['Sig']

    # Extract model parameters.
    N = sde_struct['N']
    D = sde_struct['D']

    # Time discretization step.
    dt = sde_struct['dt']

    # Total number of linear parameters 'At'.
    K = D*D*N

    # Unpack data.
    if (D == 1):
        A = Ab[:N]
        b = Ab[N:]
    else:
        A = Ab[:K].reshape(N,D,D)
        b = Ab[K:].reshape(N,D)
    # ...

    # Extract the initial (posterior) moments.
    m0 = mParam['m0']
    S0 = mParam['S0']

    # Model specific energy function.
    fun = mParam['fun']

    # Forward sweep to get consistent 'm' and 'S'.
    m, S = fwd_ode.solver(A, b, m0, S0, Sigma, dt, ode_method)

    # Update the structure with the (new)
    # posterior moments {m(t), S(t).
    mParam['mt'] = m
    mParam['St'] = S

    # Energy from the observations (Likelihood).
    # NB: unused parameters: dEobs_dR
    Eobs, dEobs_dm, dEobs_dS, _ = likelihood.gauss(m, S, sde_struct)

    # Energy from the sde.
    # NB: unused parameters: dEsde_dth, dEsde_dSig
    Esde, Efx, Edf, dEsde_dm, dEsde_dS, _, _ = fun(A, b, m, S, sde_struct)

    # Expectations of Esde.
    mParam['Efx'] = Efx
    mParam['Edf'] = Edf

    # Backward sweep to ensure constraints are satisfied.
    lam, Psi = bwd_ode.solver(A, dEsde_dm, dEsde_dS,\
                                 dEobs_dm, dEobs_dS, dt, ode_method)

    # Lagrange multipliers: lam(t), Psi(t).
    mParam['lamt'] = lam
    mParam['Psit'] = Psi

    # Energy from the initial moment.
    # NB: unused parameters: dKL0_dm0, dKL0_dS0
    KL0, _, _ = kl0.gauss(m0, S0, lam, Psi, sde_struct)

    # Variational Free Energy (cost function value).
    F = asscalar(KL0 + Eobs + Esde)

    # --->
    return F

# End-Of-File
