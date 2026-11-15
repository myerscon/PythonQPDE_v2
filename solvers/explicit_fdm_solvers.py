"""
Explicit finite difference method solver implementations.

This module provides explicit time stepping schemes for solving partial differential
equations using finite difference spatial discretization. Includes both non-vectorized
and vectorized implementations for different performance requirements.

The explicit solvers use forward time stepping where the solution at the next time
step is computed directly from known values at the current time step.
"""

import numpy as np
import kacewicz_functions
from solvers.simulation import Simulation
from sympy_files.fdm_equation_writer import Explicit_fdm_equation_writer

class Explicit_fdm_simulation(Simulation):
    """ Non-vectorized explicit finite difference simulation.
    
    Implements explicit time stepping where derivatives are computed directly
    from known values at the current time step. Uses list-based approach for
    integration over secondary intervals within each primary interval.
    
    The explicit scheme advances the solution using Taylor polynomial coefficients
    computed at each secondary interval, with integration performed point-by-point
    across the spatial grid.
    
    Attributes:
        Inherits all attributes from Simulation base class.
        
    Notes:
        Explicit schemes require time step restrictions for stability (CFL condition).
        Custom output times are handled by interpolating within secondary intervals.
        Boundary conditions are applied after each integration step.
    """
    
    def __init__(self, general_params, sympy_params):
        """ Initialize explicit finite difference simulation.
        
        Args:
            general_params (dict): Simulation parameters including grid, time stepping,
                boundary conditions, and solver options.
            sympy_params (dict): Symbolic equation definitions, auxiliary functions,
                spatial variables, and constants.
        """
        super().__init__(general_params, sympy_params)

    def _initialize_auto_sympy(self, sympy_params):
        """ Initialize symbolic equation processing for explicit scheme.
        
        Sets up the equation writer and performs finite difference approximation
        for explicit time stepping. Creates lambda functions for evaluating
        driver functions and their derivatives.
        
        Args:
            sympy_params (dict): Symbolic parameters including equations, auxiliaries,
                spatial variables, and constants.
                
        Notes:
            Uses Explicit_fdm_equation_writer to handle spatial derivative replacement
            and create evaluable lambda functions for time integration.
        """
        self.evals = Explicit_fdm_equation_writer(
            sympy_params['list_equations'],
            sympy_params['list_auxiliary'],
            sympy_params['list_spatial_vars'],
            sympy_params['list_constants'],
            sympy_params['temporal_var'],
            self.r,
            self.index_range,
            self.factorials
        )
        self.evals.finite_diff_approx(z=sympy_params['list_spatial_vars'][0], dz_sym=sympy_params['list_constants'][0][0])

    def IntegrateGij(self, t_start, primary_interval):
        """ Integrate PDEs over a primary interval using list-based approach.
        
        Performs time integration over all secondary intervals within the current
        primary interval. For each secondary interval and spatial point, evaluates
        the integrand function using stored Taylor polynomial coefficients.
        
        Args:
            t_start (float): Starting time of the current primary interval.
            primary_interval (int): Index of the current primary interval.
            
        Notes:
            Uses non-vectorized integration where each spatial point is processed
            individually. Handles custom output times by checking for matches
            within each secondary interval and performing partial integration.
            Updates boundary conditions after completing each equation.
        """
        # Pre-compute some values
        N_range = range(self.N - 2)
        M_range = range(self.n**(self.k-1))
        
        for k in range(self.num_ODEs):
            sums = np.zeros(self.N - 2)
            for j in M_range:
                t_old = t_start + (j * self.hbar)
                t_new = t_start + ((j + 1) * self.hbar)
                
                # Prepare integration parameters
                C = self.ll[k, 1:, :, j].T
                t0 = np.full(self.N - 2, t_old)
                
                # Perform integration for each spatial point
                integration_results = np.array([
                    kacewicz_functions.integrate(
                        integrand_function=self.integrand_func,
                        tlims=np.array([t_old, t_new]),
                        n_samples=self.n_samples,
                        integral_mode=self.integral_mode,
                        C=C[i],
                        t0=t0[i],
                        quantum=self.quantum,
                        M=self.M,
                        delta=self.delta,
                        use_vectorized=False
                    ) for i in N_range
                ])
                
                sums += integration_results
                
                # Handle custom output times
                if self.t_counter < self.t_output_length:
                    custom_time = self.t_outputs[self.t_counter]
                    if t_old <= custom_time < t_new:
                        if custom_time == t_old:
                            self.custom_outputs[k, 1:-1, self.t_counter] = self.y[k, 1:-1, primary_interval]
                        else:
                            custom_integration = np.array([
                                kacewicz_functions.integrate(
                                    integrand_function=self.integrand_func,
                                    tlims=np.array([t_old, custom_time]),
                                    n_samples=self.n_samples,
                                    integral_mode=self.integral_mode,
                                    C=C[i],
                                    t0=t0[i],
                                    quantum=self.quantum,
                                    M=self.M,
                                    delta=self.delta,
                                    use_vectorized=False
                                ) for i in N_range
                            ])
                            self.custom_outputs[k, 1:-1, self.t_counter] = self.y[k, 1:-1, primary_interval] + custom_integration
                        
                        if k == self.num_ODEs - 1:
                            for i in range(self.num_ODEs):
                                self.set_boundary_values(self.custom_outputs, i, time_index=self.t_counter)
                            self.update_save_custom_h5(self.t_counter)
                            self.t_counter += 1
            
            self.y[k, 1:-1, primary_interval + 1] = self.y[k, 1:-1, primary_interval] + sums
            
            if self.verbose:
                print(f"Completed equation {k + 1} out of {self.num_ODEs}...")
        
        # Set boundary values for the final state
        for k in range(self.num_ODEs):
            current_time = t_start + self.hbar * self.n**(self.k-1)
            self.set_boundary_values(self.y, index=k, time_index=primary_interval + 1, current_time=current_time)

class Explicit_vectorized_fdm_simulation(Simulation):
    """ Vectorized explicit finite difference simulation.
    
    Implements explicit time stepping with vectorized operations for improved
    computational efficiency. Processes multiple spatial points simultaneously
    during integration over secondary intervals.
    
    The vectorized approach reduces function call overhead and enables better
    utilization of NumPy's optimized array operations compared to the
    non-vectorized implementation.
    
    Attributes:
        Inherits all attributes from Simulation base class.
        
    Notes:
        Provides same functionality as non-vectorized version but with better
        performance for larger spatial grids. Memory usage may be higher due
        to vectorized array operations.
    """
    
    def __init__(self, general_params, sympy_params):
        """ Initialize vectorized explicit finite difference simulation.
        
        Args:
            general_params (dict): Simulation parameters including grid, time stepping,
                boundary conditions, and solver options.
            sympy_params (dict): Symbolic equation definitions, auxiliary functions,
                spatial variables, and constants.
        """
        super().__init__(general_params, sympy_params)

    def _initialize_auto_sympy(self, sympy_params):
        """ Initialize symbolic equation processing for vectorized explicit scheme.
        
        Sets up the equation writer and performs finite difference approximation
        for explicit time stepping with vectorized evaluation support.
        
        Args:
            sympy_params (dict): Symbolic parameters including equations, auxiliaries,
                spatial variables, and constants.
                
        Notes:
            Uses same Explicit_fdm_equation_writer as non-vectorized version.
            The vectorization occurs during integration, not in symbolic processing.
        """
        self.evals = Explicit_fdm_equation_writer(
            sympy_params['list_equations'],
            sympy_params['list_auxiliary'],
            sympy_params['list_spatial_vars'],
            sympy_params['list_constants'],
            sympy_params['temporal_var'],
            self.r,
            self.index_range,
            self.factorials
        )
        self.evals.finite_diff_approx(z=sympy_params['list_spatial_vars'][0], dz_sym=sympy_params['list_constants'][0][0])

    def IntegrateGij(self, t_start, primary_interval):
        """ Integrate PDEs over a primary interval using vectorized operations.
        
        Performs time integration over all secondary intervals within the current
        primary interval using vectorized array operations. Processes all interior
        spatial points simultaneously for each secondary interval.
        
        Args:
            t_start (float): Starting time of the current primary interval.
            primary_interval (int): Index of the current primary interval.
            
        Notes:
            Prepares vectorized input arrays (time limits, coefficients, initial times)
            and calls vectorized integration functions. Significantly faster than
            non-vectorized version for problems with many spatial grid points.
            Custom output times handled through vectorized partial integration.
        """
        N_range = self.N - 2
        M_range = self.n**(self.k-1)
        
        for k in range(self.num_ODEs):
            sums = np.zeros(N_range)
            for j in range(M_range):
                t_old = t_start + (j * self.hbar)
                t_new = t_start + ((j + 1) * self.hbar)
                
                # Prepare integration parameters
                tlims = np.column_stack((np.full(N_range, t_old), np.full(N_range, t_new)))
                C = self.ll[k, 1:, :, j].T
                t0 = np.full(N_range, t_old)
                
                # Perform vectorized integration
                integration_results = kacewicz_functions.integrate(
                    integrand_function=self.integrand_func,
                    tlims=tlims,
                    n_samples=self.n_samples,
                    integral_mode=self.integral_mode,
                    C=C,
                    t0=t0,
                    quantum=self.quantum,
                    M=self.M,
                    delta=self.delta,
                    use_vectorized=True
                )
                
                sums += integration_results
                
                # Handle custom output times
                if self.t_counter < self.t_output_length:
                    custom_time = self.t_outputs[self.t_counter]
                    if t_old <= custom_time < t_new:
                        if custom_time == t_old:
                            self.custom_outputs[k, 1:-1, self.t_counter] = self.y[k, 1:-1, primary_interval]
                        else:
                            custom_tlims = np.column_stack((np.full(N_range, t_old), np.full(N_range, custom_time)))
                            custom_integration = kacewicz_functions.integrate(
                                integrand_function=self.integrand_func,
                                tlims=custom_tlims,
                                n_samples=self.n_samples,
                                integral_mode=self.integral_mode,
                                C=C,
                                t0=t0,
                                quantum=False,
                                M=self.M,
                                delta=self.delta,
                                use_vectorized=True
                            )
                            self.custom_outputs[k, 1:-1, self.t_counter] = self.y[k, 1:-1, primary_interval] + custom_integration
                        
                        if k == self.num_ODEs - 1:
                            for i in range(self.num_ODEs):
                                self.set_boundary_values(self.custom_outputs, i, time_index=self.t_counter)
                            self.update_save_custom_h5(self.t_counter)
                            self.t_counter += 1
            
            self.y[k, 1:-1, primary_interval + 1] = self.y[k, 1:-1, primary_interval] + sums
            
            if self.verbose:
                print(f"Completed equation {k + 1} out of {self.num_ODEs}...")
        
        # Set boundary values for the final state
        for k in range(self.num_ODEs):
            current_time = t_start + self.hbar * self.n**(self.k-1)
            self.set_boundary_values(self.y, index=k, time_index=primary_interval + 1, current_time=current_time)