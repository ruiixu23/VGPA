
# coding: utf-8

# In[ ]:


# Get the system environment variables
import os
import pickle
import sys

# General Imports
import numpy as np
from numpy import linalg
from time import time
# from IPython import display

# Variational
from auxiliary import collect_obs
from auxiliary import initialize_Ab0
from core import smoothing

# Plotting
# from matplotlib import cm
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Dynamic
from dynamics import sys_lorenz_63 as dynamics

# get_ipython().run_line_magic('matplotlib', 'inline')

# Change the default figure size.
# plt.rcParams['figure.figsize'] = (10.0, 8.0)


# In[ ]:


def get_sample_path_file_name(base_dir):
    return '{}/sample.pickle'.format(base_dir)


# In[ ]:


def generate_sample_path(base_dir):
    '''
    TIME-WINDOW PARAMETERS:
    '''
    # Initial, final and time step.
    t0 = 0
    tf = 10
    dt = 0.01

    # Define the time-window of inference.
    Tw = np.arange(t0, tf + dt, dt)

    # Number of discretized points.
    N = Tw.shape[0]

    '''
    SYSTEM SPECIFIC PARAMETERS:
    '''
    # Dimensionality of the system.
    D = 3

    # Stochastic Noise (variance).
    sigma_Noise = 10 * np.eye(D)

    # Drift parameters (sigma, rho, beta).
    theta_Drift = np.array([10, 28, 8. / 3])

    # Observation Noise (variance).
    obs_Noise = 2 * np.eye(D)

    # Define the observation density (# of observations per time unit).
    n_Obs = 5

    # We need at least one observation (per time unit)
    n_Obs = np.max([n_Obs, 1])

    # Observation operator: np.eye(D)
    H = np.eye(D)

    # Create the (artificial) true trajectory.
    xt_true = dynamics.system_path(Tw, sigma_Noise, theta_Drift)

    # Sample the noisy observations from the true path.
    obsX, obsY = collect_obs.collect_obs(xt_true, Tw, n_Obs, obs_Noise)

    data = {
        't0': t0,
        'tf': tf,
        'dt': dt,
        'Tw': Tw,
        'N': N,
        'D': D,
        'sigma_Noise': sigma_Noise,
        'theta_Drift': theta_Drift,
        'obs_Noise': obs_Noise,
        'n_Obs': n_Obs,
        'H': H,
        'xt_true': xt_true,
        'obsX': obsX,
        'obsY': obsY
    }

    file_name = get_sample_path_file_name(base_dir)

    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

    return data


# In[ ]:


def load_sample_path(base_dir):
    file_name = get_sample_path_file_name(base_dir)
    with open(file_name, 'rb') as file:
        return pickle.load(file)


# In[ ]:


def get_result_file_name(base_dir, it):
    return '{}/result-{}.pickle'.format(base_dir, it)


# In[ ]:


def run_smoothing(data, theta_Drift, sigma_Noise):
    # Define the time-window of inference.
    Tw = data['Tw']

    D = data['D']

    # Prior moment of initial condition noise variance.
    # p(x0) ~ N(mu, tau0)
    tau0 = 0.5 * np.eye(D)

    # Get the true sample value at time t=0
    prior_x0 = {'mu0': data['xt_true'][0], 'tau0': tau0}

    # Initial mean m(t=0)
    m0 = data['xt_true'][0] + 0.1 * np.random.randn(1)

    # Initial covariance matrix S(t=0): K*np.eye(D)
    S0 = 0.25 * np.eye(D)

    # ODE solver: {'Euler', 'Heun', 'RK2', 'RK4'}
    ode_method = 'Euler'

    sde_struct = {
        'Sig': sigma_Noise,
        'theta': theta_Drift,
        'Rig': data['obs_Noise'],
        'D': D,
        'H': data['H'],
        'obsX': data['obsX'],
        'obsY': data['obsY'],
        'px0': prior_x0,
        'Tw': Tw,
        'dt': data['dt'],
        'N': data['N'],
        'ode_method': ode_method,
        'checkGradf': False
    }

    # Generate initial variational parameters (initial search point).
    Ab0 = initialize_Ab0.initialize_Ab0(S0, sde_struct)

    # Main Operation.
    print('[VGPA] (Smoothing) Experiment in progress. Please wait ...')

    # Full Variational approximation.
    Fmin, mParam = smoothing.smoothing(dynamics.energy_mode, Ab0, m0, S0, sde_struct)

    return Fmin, mParam, sde_struct


# In[ ]:


def estimate_paramters(base_dir, theta_Drift, sigma_Noise, rate=0.01, nit=250):
    costs = []

    # Start tracking time
    tic = time()

    # Inner loop
    data = load_sample_path(base_dir)
    Fmin, mParam, sde_struct = run_smoothing(data, theta_Drift, sigma_Noise)
    costs.append(Fmin)

    # display.clear_output()

    it = 0

    Fmin_old = Fmin
    theta_Drift_old = theta_Drift.copy()

    while it < nit:
        # Calculate gradient
        _, _, _, _, _, dEsde_dth, _ = dynamics.energy_mode(mParam['At'], mParam['bt'], mParam['mt'], mParam['St'], sde_struct)

        if not np.any(dEsde_dth):
            return

        # Update parameter values
        theta_Drift -= rate * dEsde_dth

        # Inner loop
        data = load_sample_path(base_dir)
        Fmin, mParam, sde_struct = run_smoothing(data, theta_Drift, sigma_Noise)
        costs.append(Fmin)

        # display.clear_output()

        # Calculate change in free energy and parameters
        Fmin_diff = abs(Fmin - Fmin_old)
        theta_Drift_diff = linalg.norm(theta_Drift - theta_Drift_old, 2)

        Fmin_old = Fmin
        theta_Drift_old = theta_Drift.copy()

        ttime = time() - tic

        print('Current Iteration:')
        print(it)
        print()

        print('Runtime:')
        print(ttime)
        print()

        print('Costs:')
        print(', '.join(['{:.2f}'.format(cost) for cost in costs]))
        print()

        print('Gradients:')
        print(dEsde_dth)
        print()

        print('Changes:')
        print(Fmin_diff, theta_Drift_diff)
        print()

        print('Parameter:')
        print(', '.join(['{:.2f}'.format(item) for item in theta_Drift]))
        print()

        # Save each iteration
        with open(get_result_file_name(base_dir, it), 'wb') as file:
            pickle.dump({
                'theta_Drift': theta_Drift,
                'sigma_Noise': sigma_Noise,
                'Fmin': Fmin,
                'mParam': mParam,
                'costs': costs,
                'gradients': dEsde_dth,
                'changes': [Fmin_diff, theta_Drift_diff],
                'it': it,
                'time': ttime
            }, file)

        if Fmin_diff <= 1e-6 or theta_Drift_diff <= 1e-6:
            return

        # Increament iteration count
        it += 1

    print('Maximum iteration reached')


# In[ ]:


try:
    base_dir = './results/parameter-lorenz-63/{}'.format(sys.argv[1])
    rate = float(sys.argv[2])
except:
    base_dir = './results/parameter-lorenz-63'
    rate = 0.001

print('Base dir: {}'.format(base_dir))
print('Learning rate: {}'.format(rate))

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

if not os.path.exists(get_sample_path_file_name(base_dir)):
    print('Generating new sample path')
    generate_sample_path(base_dir)

theta_Drift = np.array([0.0, 0.0, 0.0])
sigma_Noise = 10 * np.eye(3)

estimate_paramters(base_dir, theta_Drift, sigma_Noise, rate=0.001)
