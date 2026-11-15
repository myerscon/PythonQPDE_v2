"""
kacewicz_functions.py

Utility functions for quantum integration and parameter calculations used in the 
Python Quantum Partial Differential Equation (QPDE) solver.

This module provides functions for:
- Time partitioning parameter calculations for hierarchical time stepping
- Classical and quantum numerical integration methods (Riemann and Gauss quadrature)
- Quantum Amplitude Estimation (QAE) algorithms for quantum computation modeling

The integration subroutines are adapted from algorithms developed by Frank Gaitan.
"""

import numpy as np
from scipy.special import roots_legendre

# Time partitioning functions:

def calculate_k(epsilon, n):
    """
    Calculate the secondary interval parameter k from error tolerance.

    Computes the number of hierarchical levels needed to achieve the specified
    error tolerance using the relationship: epsilon = 1 / (n^(k-1))

    Args:
        epsilon (float): User-specified error tolerance (0 < epsilon < 1).
        n (int): Number of primary intervals (n >= 2).

    Returns:
        int: Number of hierarchical levels k, where total secondary intervals
            per primary = n^(k-1).
    """
    return int(np.ceil(1 - np.log(epsilon) / np.log(n)))

def calculate_partition_parameters_explicit(epsilon, M, n0=2):
    """
    Calculate partition parameters for explicit time discretization schemes.

    Iteratively determines the optimal number of primary intervals (n) and
    hierarchical levels (k) to satisfy both error tolerance and total interval
    constraints.

    Args:
        epsilon (float): Maximum error tolerance between approximation and solution.
        M (int): Total number of secondary intervals in the time domain.
        n0 (int, optional): Initial guess for number of primary intervals. 
            Defaults to 2.

    Returns:
        tuple: (n, k) where:
            - n (int): Number of primary intervals
            - k (int): Hierarchical levels (secondary intervals per primary = n^(k-1))

    Notes:
        The algorithm ensures n^k >= M while maintaining the error bound
        epsilon = 1 / (n^(k-1)).
    """
    n = n0
    k = calculate_k(epsilon, n)

    while (n ** k < M):
        n += 1
        k = calculate_k(epsilon, n)

    return (n, k)

def calculate_partition_parameters_implicit(epsilon, T, hbar):
    """
    Calculate partition parameters for implicit time discretization schemes.

    Determines the number of primary intervals and hierarchical levels based on
    total simulation time and time step size.

    Args:
        epsilon (float): Maximum error tolerance between approximation and solution.
        T (float): Total simulation time.
        hbar (float): Time step size.

    Returns:
        tuple: (n, k) where:
            - n (int): Number of primary intervals  
            - k (int): Hierarchical levels

    Notes:
        For implicit schemes: n = ceil(epsilon*T/hbar), k = ceil(log(T/hbar)/log(n))
    """
    n = int(np.ceil(epsilon*T/hbar))
    k = int(np.ceil(np.log(T/hbar)/np.log(n)))
    return (n,k)

def calculate_partition_parameters(time_discretization, epsilon, time_disc_params):
    """
    Unified interface for calculating time partition parameters.

    Dispatches to appropriate parameter calculation function based on the
    specified time discretization scheme.

    Args:
        time_discretization (str): Time discretization scheme type.
            Must be 'explicit' or 'implicit'.
        epsilon (float): Maximum error tolerance between approximation and solution.
        time_disc_params (list): Parameters specific to discretization method:
            - For 'explicit': [M, n0] or [M] where M is total intervals, 
              n0 is optional initial guess
            - For 'implicit': [T, hbar] where T is total time, hbar is time step

    Returns:
        tuple: (n, k) partition parameters.

    Raises:
        ValueError: If time_discretization is not 'explicit' or 'implicit'.
    """
    if (time_discretization == 'explicit'):
        if (len(time_disc_params)>1):
            return calculate_partition_parameters_explicit(epsilon,time_disc_params[0],time_disc_params[1])
        else:
            return calculate_partition_parameters_explicit(epsilon,time_disc_params[0])
    elif (time_discretization == 'implicit'):
        return calculate_partition_parameters_implicit(epsilon,time_disc_params[0],time_disc_params[1])
    else:
        print("ERROR: Invalid time_discretization variable provided!")

# Quantum integration functions:

def integrate_riemann(integrand_function, tlims, n_samples, C, t0, delta, M, quantum=False):
    """
    Perform left Riemann sum integration with optional quantum computation modeling.

    Computes the integral using left Riemann sum approximation. When quantum=True,
    models how a quantum computer would solve the same problem using quantum
    amplitude estimation.

    Args:
        integrand_function (callable): Time-dependent integrand function that accepts
            batched data with signature f(t, t0, C).
        tlims (tuple): Integration bounds (t_lower, t_upper).
        n_samples (int): Number of sample points for Riemann sum.
        C (np.ndarray): Coefficients for the integrand function.
        t0 (float): Initial time parameter for integrand.
        delta (float): Probability that quantum integration result violates 
            error bounds (used only if quantum=True).
        M (int): Dimension of Hilbert space for quantum estimation.
        quantum (bool, optional): Whether to model quantum computation. 
            Defaults to False.

    Returns:
        float: Computed integral value.

    Notes:
        Classical mode (quantum=False) is faster for testing and debugging.
        Quantum mode (quantum=True) models quantum computation but is slower
        on classical computers.
    """
    # Generate evenly spaced points for Riemann sum
    t = np.linspace(tlims[0], tlims[1], n_samples, endpoint=False)
    f = integrand_function(t, t0, C)

    if not quantum:
        return (tlims[1] - tlims[0]) * np.mean(f)

    f_max, f_min = np.max(f), np.min(f)
    f_diff = f_max - f_min
    tolerance = 1e-12

    if f_diff > tolerance:
        # Scale function values for quantum estimation
        f_scaled = (f - f_min) / f_diff
        f_avg = np.mean(f_scaled)
        omega = np.arcsin(np.sqrt(f_avg)) / np.pi
        estimates = QAmpEst(M, omega, delta)
    else:
        estimates = 0.0

    return (tlims[1] - tlims[0]) * (estimates * f_diff + f_min)

def integrate_riemann_vectorized(integrand_function, tlims, n_samples, C, t0, delta, M, quantum=False):
    """
    Perform vectorized left Riemann sum integration.

    Vectorized version of Riemann integration that processes multiple integration
    problems simultaneously for improved computational efficiency.

    Args:
        integrand_function (callable): Time-dependent integrand function for batched data.
        tlims (np.ndarray): Integration bounds array of shape (N, 2) where each row
            contains (t_lower, t_upper) for one integration problem.
        n_samples (int): Number of sample points for each Riemann sum.
        C (np.ndarray): Coefficient array for integrand functions.
        t0 (np.ndarray): Initial time parameters array.
        delta (float): Quantum error probability threshold.
        M (int): Hilbert space dimension for quantum estimation.
        quantum (bool, optional): Model quantum computation. Defaults to False.

    Returns:
        np.ndarray: Array of computed integral values, one per integration problem.
    """
    dt = (tlims[:, 1] - tlims[:, 0]) / n_samples
    # Generate evenly spaced points for Riemann sum
    t = tlims[:, 0][:, np.newaxis] + dt[:, np.newaxis] * np.arange(n_samples)
    
    f = integrand_function(t, t0[:, np.newaxis], C)
    
    if not quantum:
        return np.sum(f * dt[:, np.newaxis], axis=1)
    
    f_max, f_min = np.max(f, axis=1), np.min(f, axis=1)
    f_diff = np.maximum(f_max - f_min, 1e-12)
    
    # Scale function values for quantum estimation
    f_scaled = np.where(f_diff[:, np.newaxis] > 1e-12,
                        (f - f_min[:, np.newaxis]) / f_diff[:, np.newaxis],
                        0.5)
    
    f_avg = np.mean(f_scaled, axis=1)
    omega = np.arcsin(np.sqrt(np.clip(f_avg, 0, 1))) / np.pi
    estimates = QAmpEst_vectorized(M, omega, delta)
    
    result = n_samples * (estimates * f_diff + f_min) * dt
    return np.where(np.isfinite(result), result, 0.0)

def integrate_quad(integrand_function, tlims, n_samples, C, t0, delta, M, quantum=False):
    """
    Perform Gauss-Legendre quadrature integration.

    High-accuracy integration using Gauss-Legendre quadrature nodes and weights,
    with optional quantum computation modeling.

    Args:
        integrand_function (callable): Time-dependent integrand function.
        tlims (tuple): Integration bounds (t_lower, t_upper).
        n_samples (int): Number of quadrature points (determines accuracy order).
        C (np.ndarray): Coefficients for the integrand function.
        t0 (float): Initial time parameter.
        delta (float): Quantum error probability threshold.
        M (int): Hilbert space dimension for quantum estimation.
        quantum (bool, optional): Model quantum computation. Defaults to False.

    Returns:
        float: Computed integral value.

    Notes:
        Gauss-Legendre quadrature provides 2n-1 degree polynomial exactness
        with n points. Includes coordinate transformation from [-1,1] to integration interval.
    """
    if np.isclose(tlims[1], tlims[0]):
        #print(f"Warning: Integration limits are too close: {tlims}")
        return 0.0
    
    # Calculate parameters for Gauss-Legendre quadrature
    a = 2 / (tlims[1] - tlims[0])
    b = 1 - tlims[1] * a
    roots, weights = roots_legendre(n_samples)
    t = (roots - b) / a  # Transform roots to integration interval
    y = integrand_function(t, t0, C)
    f = y * weights
    
    if not quantum:
        return np.sum(f) / a
    
    f_max, f_min = np.max(f), np.min(f)
    f_diff = f_max - f_min
    tolerance = 1e-12
    
    if f_diff > tolerance:
        # Scale function values for quantum estimation
        f_scaled = (f - f_min) / f_diff
        f_avg = np.mean(f_scaled)
        omega = np.arcsin(np.sqrt(np.clip(f_avg, 0, 1))) / np.pi
        estimates = QAmpEst(M, omega, delta)
    else:
        estimates = 0.0
    
    result = n_samples * (estimates * f_diff + f_min) / a
    if not np.isfinite(result):
        print(f"Warning: Non-finite result in integrate_quad. tlims: {tlims}, n_samples: {n_samples}")
        return 0.0
    return result

def integrate_quad_vectorized(integrand_function, tlims, n_samples, C, t0, delta, M, quantum=False):
    """
    Perform vectorized Gauss-Legendre quadrature integration.

    Vectorized implementation of Gauss quadrature that processes multiple
    integration problems simultaneously with shared quadrature points and weights.

    Args:
        integrand_function (callable): Time-dependent integrand function for batched data.
        tlims (np.ndarray): Integration bounds array of shape (N, 2).
        n_samples (int): Number of quadrature points per integration.
        C (np.ndarray): Coefficient array for integrand functions.
        t0 (np.ndarray): Initial time parameters array.
        delta (float): Quantum error probability threshold.
        M (int): Hilbert space dimension for quantum estimation.
        quantum (bool, optional): Model quantum computation. Defaults to False.

    Returns:
        np.ndarray: Array of computed integral values.
    """
    # Calculate parameters for Gauss-Legendre quadrature
    a = 2 / (tlims[:, 1] - tlims[:, 0])
    b = 1 - tlims[:, 1] * a
    roots, weights = roots_legendre(n_samples)
    
    # Transform roots to integration intervals
    t = ((roots.reshape(1, -1) - b.reshape(-1, 1)) / a.reshape(-1, 1))
    y = integrand_function(t, t0.reshape(-1, 1), C)
    f = y * weights
    
    if not quantum:
        return np.sum(f, axis=1) / a
    
    f_max, f_min = np.max(f, axis=1), np.min(f, axis=1)
    f_diff = f_max - f_min
    tolerance = 1e-12
    
    f_diff_safe = np.maximum(f_diff, tolerance)
    # Scale function values for quantum estimation
    f_scaled = np.where(f_diff[:, np.newaxis] > tolerance, 
                        (f - f_min[:, np.newaxis]) / f_diff_safe[:, np.newaxis], 
                        0.5)
    
    f_avg = np.mean(f_scaled, axis=1)
    omega = np.arcsin(np.sqrt(np.clip(f_avg, 0, 1))) / np.pi
    estimates = QAmpEst_vectorized(M, omega, delta)
    
    result = n_samples * (estimates * f_diff + f_min) / a
    return np.where(np.isfinite(result), result, 0.0)

def integrate(integrand_function, tlims, n_samples, C, t0, delta, M, integral_mode="quad", quantum=False, use_vectorized=True):
    """
    Unified integration interface supporting multiple methods and modes.

    Provides a single entry point for integration with automatic dispatch to
    appropriate method (Riemann vs quadrature) and mode (vectorized vs scalar).

    Args:
        integrand_function (callable): Time-dependent integrand function.
        tlims (tuple or np.ndarray): Integration bounds. Tuple (a,b) for scalar,
            array of shape (N,2) for vectorized.
        n_samples (int): Number of integration sample points.
        integral_mode (str, optional): Integration method - "quad" for Gauss-Legendre
            quadrature or "riemann" for Riemann sum. Defaults to "quad".
        C (np.ndarray): Coefficients for the integrand function.
        t0 (float or np.ndarray): Initial time parameter(s).
        delta (float): Quantum error probability threshold.
        M (int): Hilbert space dimension for quantum estimation.
        quantum (bool, optional): Model quantum computation. Defaults to False.
        use_vectorized (bool, optional): Use vectorized implementation when possible.
            Defaults to True.

    Returns:
        float or np.ndarray: Computed integral value(s).

    Raises:
        ValueError: If integral_mode is not "quad" or "riemann".
    """
    if integral_mode == "quad":
        if use_vectorized:
            return integrate_quad_vectorized(integrand_function, tlims, n_samples, C=C, t0=t0, quantum=quantum, delta=delta, M=M)
        else:
            return integrate_quad(integrand_function, tlims, n_samples, C=C, t0=t0, quantum=quantum, delta=delta, M=M)
    elif integral_mode == "riemann":
        if use_vectorized:
            return integrate_riemann_vectorized(integrand_function, tlims, n_samples, C=C, t0=t0, quantum=quantum, delta=delta, M=M)
        else:
            return integrate_riemann(integrand_function, tlims, n_samples, C=C, t0=t0, quantum=quantum, delta=delta, M=M)
    else:
        raise ValueError("kacewicz integrate() - Invalid integral_mode")
    
# Non-Vectorized QAmpEst function:

def QAmpEst(M, omega, delta):
    """
    Estimate quantum amplitude using Quantum Amplitude Estimation algorithm.

    Implements the quantum amplitude estimation algorithm based on Brassard et al.
    to model quantum computation of unknown quantum amplitudes.

    Args:
        M (int): Dimension of the Hilbert space.
        omega (float): Unknown scalar parameter for quantum amplitude derivation
            (0 <= omega <= 1).
        delta (float): One minus the probability that amplitude estimation is
            within error bounds.

    Returns:
        float: Estimate of the quantum amplitude (sin²(πω)).

    Notes:
        Based on Brassard et al. (quant-ph/0005055).
        Uses median of multiple random estimates for robustness.
        Original implementation by F. Gaitan.
    """
    # Calculate the number of runs needed for the desired confidence level
    total_runs = int(np.ceil(-8 * np.log(delta)))
    if total_runs % 2 == 0:
        total_runs += 1  # Ensure odd number of runs for a unique median

    # Generate estimates using randQAEA
    estimates = np.array([randQAEA(M, omega) for _ in range(total_runs)])
    median_estimate = np.median(estimates)
    # Convert the median estimate to an amplitude estimate
    estimate_amplitude = np.sin(np.pi * median_estimate / M) ** 2

    return estimate_amplitude

def randQAEA(M, omega):
    """
    Generate random deviate for Quantum Amplitude Estimation Algorithm.

    Samples a random deviate from the normalized sinc²(x) probability distribution
    over discrete values, as used in quantum amplitude estimation.

    Args:
        M (int): Dimension of the Hilbert space.
        omega (float): Unknown scalar parameter (0 <= omega <= 1).

    Returns:
        int: Random deviate sampled from the discrete sinc² distribution.

    Notes:
        Based on Brassard et al. (quant-ph/0005055).
        Original implementation by F. Gaitan.
    """
    Momega = M * omega
    y = np.arange(M)
    x = y - Momega  # Shifted domain appearing in p(y)
    
    tiny = 1e-24  # Small value to avoid division by zero
    # Calculate probabilities
    tempProb = np.sin(np.pi * (x + tiny)) / np.sin(np.pi * (x + tiny) / M)
    Probs = (1 / M ** 2) * (tempProb ** 2)
    
    p_sums = np.cumsum(Probs)  # Partial sums of probabilities
    u = np.random.random()  # Generate a random sample
    
    # Find the subinterval containing the random sample
    return np.argmax(p_sums >= u)

# Vectorized QAmpEst function:

def QAmpEst_vectorized(M, omega, delta):
    """
    Vectorized Quantum Amplitude Estimation for multiple omega values.

    Efficiently computes quantum amplitude estimates for arrays of omega parameters
    using vectorized operations.

    Args:
        M (int): Dimension of the Hilbert space.
        omega (np.ndarray): Array of unknown scalar parameters (each: 0 <= omega[i] <= 1).
        delta (float): Confidence parameter.

    Returns:
        np.ndarray: Array of quantum amplitude estimates.
    """
    # Calculate the number of runs needed for the desired confidence level
    total_runs = int(np.ceil(-8 * np.log(delta)))
    if total_runs % 2 == 0:
        total_runs += 1  # Ensure odd number of runs for a unique median
    
    Momega = M * omega[:, np.newaxis]
    y = np.arange(M)
    x = y[np.newaxis, :] - Momega  # Shifted domain appearing in p(y)
    tiny = 1e-24  # Small value to avoid division by zero
    # Calculate probabilities using vectorized operations
    tempProb = np.sin(np.pi * (x + tiny)) / np.sin(np.pi * (x + tiny) / M)
    Probs = (1 / M ** 2) * (tempProb ** 2)
    p_sums = np.cumsum(Probs, axis=1)  # Partial sums of probabilities
    
    # Generate random samples and find corresponding estimates
    u = np.random.random((omega.shape[0], total_runs))
    estimates = np.argmax(u[:, :, np.newaxis] <= p_sums[:, np.newaxis, :], axis=2)
    median_estimate = np.median(estimates, axis=1)
    # Convert median estimates to amplitude estimates
    return np.sin(np.pi * median_estimate / M) ** 2